"""Resumable TokCom reconstruction and perceptual-quality evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from context_maskgit_v3 import load_context_maskgit_v3
from pretrained_vq import PretrainedVQGAN
from tokcom_arq import transmit_tokens_with_arq
from tokcom_image_metrics import TokComImageMetrics, psnr_per_image


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE17_PERCEPTUAL_PROTOCOL.json"
OUT = ROOT / "runs/stage17_perceptual"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist(),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_original(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def condition_key(condition: dict) -> str:
    snr = str(float(condition["snr_esn0_db"])).replace(".", "p")
    return f'{condition["channel"]}_{snr}db_seed{condition["seed"]}_{condition["mode"]}'


def validate(protocol: dict) -> dict:
    if protocol.get("status") != "frozen":
        raise ValueError("Stage-17 protocol must be frozen")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-17 implementation changed: {name}")
    for field, hash_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("validation_cache", "validation_cache_sha256"),
        ("clip_checkpoint", "clip_checkpoint_sha256"),
        ("vqgan_checkpoint", "vqgan_checkpoint_sha256"),
    ):
        if sha256(ROOT / protocol[field]) != protocol[hash_field]:
            raise ValueError(f"Frozen Stage-17 asset changed: {field}")
    stage16 = json.loads((ROOT / "STAGE16_ARQ_TCE_SUMMARY.json").read_text(encoding="utf-8"))
    if stage16.get("status") != "completed" or stage16.get("test_accessed") or stage16.get("stage14_holdout_used"):
        raise ValueError("Stage-16 completion or data-isolation audit failed")
    data = torch.load(ROOT / protocol["validation_cache"], weights_only=True)
    if len(data["ids"]) != protocol["images"]:
        raise ValueError("Frozen image count changed")
    return data


@torch.inference_mode()
def prepare_references(protocol: dict, data: dict, deadline: float) -> bool:
    directory = OUT / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    dataset_root = ROOT / protocol["dataset_root"]
    tokenizer = None
    for index, record in enumerate(data["records"]):
        target = directory / f"{index:04d}.pt"
        if target.exists():
            continue
        image_path = dataset_root / record["image"]
        if sha256(image_path) != record["sha256"]:
            raise ValueError(f"Reference image changed: {image_path}")
        if tokenizer is None:
            tokenizer = PretrainedVQGAN(ROOT / protocol["vqgan_checkpoint"])
        original = load_original(image_path)
        baseline = tokenizer.decode(data["ids"][index : index + 1])[0]
        payload = {
            "index": index,
            "sha256": record["sha256"],
            "original_uint8": original,
            "vqgan_uint8": (baseline.mul(255).round().clamp(0, 255).to(torch.uint8)),
        }
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        if time.monotonic() >= deadline:
            return False
    atomic_json(
        directory / "status.json",
        {"completed": True, "images": len(data["records"]), "test_accessed": False, "stage14_holdout_used": False},
    )
    return True


def mode_parameters(protocol: dict, condition: dict) -> tuple[dict, torch.Tensor | None, float]:
    mode = protocol["modes"][condition["mode"]]
    labels = None if mode["context"] == "none" else condition["labels"]
    return mode, labels, float(mode["guidance_scale"])


@torch.inference_mode()
def prepare_restored(protocol: dict, data: dict, condition: dict, directory: Path) -> None:
    target = directory / "restored.pt"
    communication_path = directory / "communication.json"
    if target.exists() and communication_path.exists():
        return
    receiver = load_context_maskgit_v3(torch.load(ROOT / protocol["checkpoint"], weights_only=True)).eval()
    condition = dict(condition)
    condition["labels"] = data["labels"].long()
    mode, labels, guidance = mode_parameters(protocol, condition)
    restored, missing, metrics = transmit_tokens_with_arq(
        data["ids"].numpy(),
        receiver=receiver,
        labels=labels,
        codebook_size=1024,
        snr_db=float(condition["snr_esn0_db"]),
        channel=condition["channel"],
        seed=int(condition["seed"]),
        packet_tokens=4,
        max_rounds=int(protocol["max_rounds"]),
        policy=mode["policy"],
        confidence_threshold=float(mode["confidence_threshold"]),
        recovery_steps=int(protocol["recovery_steps"]),
        guidance_scale=guidance,
    )
    communication = {
        "remaining_erasure_count": int(missing.sum()),
        "communication_metrics": metrics,
        "condition": {key: value for key, value in condition.items() if key != "labels"},
    }
    restored_tensor = torch.from_numpy(restored).to(dtype=torch.long)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(restored_tensor, temporary)
    os.replace(temporary, target)
    atomic_json(communication_path, communication)


def save_preview(path: Path, original: torch.Tensor, baseline: torch.Tensor, reconstructed: torch.Tensor) -> None:
    canvas = torch.cat((original, baseline, reconstructed), dim=2).permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(path)


@torch.inference_mode()
def evaluate_condition(protocol: dict, data: dict, condition: dict, deadline: float) -> bool:
    directory = OUT / condition_key(condition)
    metric_dir = directory / "per_image"
    preview_dir = directory / "previews"
    metric_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    prepare_restored(protocol, data, condition, directory)
    restored = torch.load(directory / "restored.pt", weights_only=True)
    communication_payload = json.loads((directory / "communication.json").read_text(encoding="utf-8"))
    tokenizer = None
    metrics = None
    preview_indices = set(protocol["preview_indices"])
    for index in range(protocol["images"]):
        target = metric_dir / f"{index:04d}.json"
        if target.exists():
            continue
        if tokenizer is None:
            tokenizer = PretrainedVQGAN(ROOT / protocol["vqgan_checkpoint"])
            metrics = TokComImageMetrics(ROOT / protocol["clip_checkpoint"], device="cpu")
        reference = torch.load(OUT / "reference" / f"{index:04d}.pt", weights_only=True)
        original = reference["original_uint8"].float().div(255)[None]
        baseline = reference["vqgan_uint8"].float().div(255)[None]
        reconstructed = tokenizer.decode(restored[index : index + 1]).cpu()
        measured = metrics.evaluate(original, reconstructed, batch_size=1)
        row = {
            "index": index,
            "sha256": data["records"][index]["sha256"],
            "token_accuracy": float((restored[index] == data["ids"][index]).float().mean()),
            "psnr": measured["per_image"]["psnr"][0],
            "clip_cosine": measured["per_image"]["clip_cosine"][0],
            "lpips": measured["per_image"]["lpips"][0],
            "vqgan_baseline_psnr": float(psnr_per_image(original, baseline)[0]),
        }
        atomic_json(target, row)
        if index in preview_indices:
            save_preview(
                preview_dir / f"{index:04d}_original_vqgan_received.png",
                reference["original_uint8"],
                reference["vqgan_uint8"],
                reconstructed[0].mul(255).round().clamp(0, 255).to(torch.uint8),
            )
        if time.monotonic() >= deadline:
            return False
    rows = [json.loads((metric_dir / f"{i:04d}.json").read_text(encoding="utf-8")) for i in range(protocol["images"])]
    communication = communication_payload["communication_metrics"]
    summary = {
        "condition": condition,
        "images": len(rows),
        "mean": {name: float(np.mean([row[name] for row in rows])) for name in ("token_accuracy", "psnr", "clip_cosine", "lpips", "vqgan_baseline_psnr")},
        "communication_metrics": communication,
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    atomic_json(directory / "summary.json", summary)
    return True


def finalize(protocol: dict) -> None:
    summaries = [json.loads((OUT / condition_key(c) / "summary.json").read_text(encoding="utf-8")) for c in protocol["conditions"]]
    grouped = []
    for channel in ("awgn", "rayleigh"):
        for mode in protocol["modes"]:
            chosen = [row for row in summaries if row["condition"]["channel"] == channel and row["condition"]["mode"] == mode]
            grouped.append({
                "channel": channel,
                "snr_esn0_db": chosen[0]["condition"]["snr_esn0_db"],
                "mode": mode,
                "seeds": len(chosen),
                **{f"{name}_mean": float(np.mean([row["mean"][name] for row in chosen])) for name in ("token_accuracy", "psnr", "clip_cosine", "lpips")},
                "tce_mean": float(np.mean([row["communication_metrics"]["tce_pixels_per_token_bit"] for row in chosen])),
                "transmissions_per_packet_mean": float(np.mean([row["communication_metrics"]["transmissions_per_packet"] for row in chosen])),
            })
    atomic_json(OUT / "summary.json", {"conditions": len(summaries), "summary": grouped, "test_accessed": False, "stage14_holdout_used": False})
    atomic_json(OUT / "status.json", {"completed": True, "conditions": len(summaries), "test_accessed": False, "stage14_holdout_used": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=220.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    torch.set_num_threads(args.threads)
    deadline = time.monotonic() + args.max_seconds
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    data = validate(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    if not prepare_references(protocol, data, deadline):
        raise SystemExit(75)
    for condition in protocol["conditions"]:
        if (OUT / condition_key(condition) / "summary.json").exists():
            continue
        if not evaluate_condition(protocol, data, condition, deadline):
            raise SystemExit(75)
        break
    if all((OUT / condition_key(c) / "summary.json").exists() for c in protocol["conditions"]):
        finalize(protocol)
        print(json.dumps({"completed": True, "conditions": len(protocol["conditions"])}))
    else:
        completed = sum((OUT / condition_key(c) / "summary.json").exists() for c in protocol["conditions"])
        print(json.dumps({"completed": False, "conditions": completed, "planned": len(protocol["conditions"])}))


if __name__ == "__main__":
    main()
