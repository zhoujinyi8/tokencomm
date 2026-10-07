"""Resumable development evaluation of Stage-13 TokCom over the explicit digital reference link."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from context_maskgit_v2 import load_context_maskgit_v2
from digital_link import transmit_tokens
from run_pretrained import spatial_fill


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "runs" / "stage13_tokcom_large"
CACHE = ROOT / "data" / "imagenette_stage10" / "token_cache" / "val.pt"
PROTOCOL = ROOT / "STAGE13_PHYSICAL_DEV_PROTOCOL.json"
OUT = ROOT / "runs" / "stage13_physical_dev"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def plan():
    return [
        {"channel": channel, "snr_esn0_db": float(snr), "seed": seed}
        for channel in ["awgn", "rayleigh"]
        for snr in [0, 4, 8, 12, 16]
        for seed in [13201, 13202, 13203]
    ]


def key(row):
    return row["channel"], float(row["snr_esn0_db"]), int(row["seed"])


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_protocol() -> dict:
    status = json.loads((TRAIN / "status.json").read_text(encoding="utf-8"))
    if not status["completed"] or status["test_accessed"]:
        raise ValueError("Stage-13 training is not complete or accessed a test set")
    best = TRAIN / "best.pt"
    cache = torch.load(CACHE, weights_only=True)
    payload = {
        "purpose": "Stage-13 development-only explicit digital-link evaluation",
        "selection_warning": "The same 150-image development split selected the checkpoint; these are not blind-test estimates.",
        "checkpoint": "runs/stage13_tokcom_large/best.pt",
        "checkpoint_sha256": sha256(best),
        "training_protocol_sha256": sha256(ROOT / "STAGE13_TOKCOM_PROTOCOL.json"),
        "validation_cache_sha256": sha256(CACHE),
        "tokenizer_sha256": cache["identity"]["tokenizer_sha256"],
        "images": len(cache["ids"]),
        "conditions": plan(),
        "link": {
            "packet_tokens": 4,
            "token_bits": 10,
            "crc": "CRC-16/CCITT-FALSE",
            "fec": "K=7 convolutional code, octal (171,133), rate 1/2, six zero termination bits",
            "modulation": "unit-average-energy Gray 16-QAM",
            "decoder": "max-log soft-input terminated Viterbi",
            "rayleigh": "iid per complex symbol with perfect receiver CSI",
            "snr_definition": "Es/N0"
        },
        "test_accessed": False,
        "used_for_training": False
    }
    atomic_json(PROTOCOL, payload)
    return payload


@torch.inference_mode()
def evaluate_one(model, data, condition):
    target = data["ids"]
    received_np, missing_np, phy = transmit_tokens(
        target.numpy(), 1024, condition["snr_esn0_db"], condition["channel"], condition["seed"], 4
    )
    received = torch.from_numpy(received_np).long()
    missing = torch.from_numpy(missing_np)
    restored_parts = []
    for start in range(0, len(target), 24):
        restored_parts.append(model.recover(received[start:start + 24].masked_fill(missing[start:start + 24], model.codes), missing[start:start + 24], steps=8))
    restored = torch.cat(restored_parts)
    spatial = torch.from_numpy(spatial_fill(received_np, missing_np)).long()
    missing_count = int(missing.sum())
    valid_received_correct = int(((~missing) & (received == target)).sum())
    result = {
        **condition,
        "images": len(target),
        "tokens": int(target.numel()),
        "missing_tokens": missing_count,
        "received_all_token_accuracy": valid_received_correct / target.numel(),
        "recovered_all_token_accuracy": float((restored == target).float().mean()),
        "maskgit_missing_accuracy": (float((restored[missing] == target[missing]).float().mean()) if missing_count else None),
        "spatial_missing_accuracy": (float((spatial[missing] == target[missing]).float().mean()) if missing_count else None),
        "undetected_wrong_tokens": int(((~missing) & (received != target)).sum()),
        "physical": phy,
        "test_accessed": False
    }
    return result


def summarize(rows):
    groups = []
    for channel in ["awgn", "rayleigh"]:
        for snr in [0, 4, 8, 12, 16]:
            chosen = [row for row in rows if row["channel"] == channel and row["snr_esn0_db"] == snr]
            groups.append({
                "channel": channel,
                "snr_esn0_db": snr,
                "seeds": len(chosen),
                "measured_per_mean": float(np.mean([row["physical"]["measured_per"] for row in chosen])),
                "measured_per_std": float(np.std([row["physical"]["measured_per"] for row in chosen])),
                "payload_ber_mean": float(np.mean([row["physical"]["payload_ber"] for row in chosen])),
                "recovered_all_token_accuracy_mean": float(np.mean([row["recovered_all_token_accuracy"] for row in chosen])),
                "recovered_all_token_accuracy_std": float(np.std([row["recovered_all_token_accuracy"] for row in chosen])),
                "maskgit_missing_accuracy_mean": (float(np.mean([row["maskgit_missing_accuracy"] for row in chosen if row["maskgit_missing_accuracy"] is not None])) if any(row["maskgit_missing_accuracy"] is not None for row in chosen) else None),
                "spatial_missing_accuracy_mean": (float(np.mean([row["spatial_missing_accuracy"] for row in chosen if row["spatial_missing_accuracy"] is not None])) if any(row["spatial_missing_accuracy"] is not None for row in chosen) else None),
                "undetected_wrong_tokens": sum(row["undetected_wrong_tokens"] for row in chosen)
            })
    return groups


def main():
    torch.set_num_threads(4)
    if not PROTOCOL.exists():
        protocol = freeze_protocol()
        print(json.dumps({"protocol_frozen": True, "conditions": len(protocol["conditions"])}))
        return
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    best_path = ROOT / protocol["checkpoint"]
    if sha256(best_path) != protocol["checkpoint_sha256"] or sha256(CACHE) != protocol["validation_cache_sha256"]:
        raise ValueError("Frozen physical-evaluation identity mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    keys = [key(row) for row in rows]
    expected = {key(row) for row in protocol["conditions"]}
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or unplanned physical result")
    pending = [condition for condition in protocol["conditions"] if key(condition) not in set(keys)]
    if not pending:
        atomic_json(OUT / "summary.json", {"conditions": len(rows), "summary": summarize(rows), "test_accessed": False, "used_for_training": False})
        atomic_json(OUT / "status.json", {"completed": True, "conditions": len(rows), "test_accessed": False})
        print(json.dumps({"completed": True, "conditions": len(rows)}))
        return
    checkpoint = torch.load(best_path, weights_only=True)
    model = load_context_maskgit_v2(checkpoint).eval()
    data = torch.load(CACHE, weights_only=True)
    row = evaluate_one(model, data, pending[0])
    rows.append(row)
    atomic_json(results_path, rows)
    print(json.dumps({"completed_condition": key(row), "count": len(rows), "planned": len(expected)}))
    if len(rows) == len(expected):
        atomic_json(OUT / "summary.json", {"conditions": len(rows), "summary": summarize(rows), "test_accessed": False, "used_for_training": False})
        atomic_json(OUT / "status.json", {"completed": True, "conditions": len(rows), "test_accessed": False})


if __name__ == "__main__":
    main()
