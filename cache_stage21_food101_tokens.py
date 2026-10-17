"""Resumably encode the Stage-20 Food-101 train/dev images into VQGAN tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from pretrained_vq import PretrainedVQGAN, WEIGHT_SHA256


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/stage20_food101"
PROTOCOL_PATH = ROOT / "STAGE21_TOKEN_CACHE_PROTOCOL.json"
OUTPUT = DATA / "token_cache_stage21"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_records(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]


def load_image(relative: str) -> torch.Tensor:
    with Image.open(DATA / relative) as image:
        fitted = ImageOps.fit(image.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS)
        array = np.asarray(fitted, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def expected_shards(rows: int, shard_size: int) -> int:
    return (rows + shard_size - 1) // shard_size


def validate_protocol(protocol: dict) -> dict:
    if protocol.get("status") != "frozen" or protocol.get("test_accessed"):
        raise ValueError("Stage-21 token-cache protocol is not safely frozen")
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-21 input changed: {name}")
    if protocol["tokenizer_sha256"] != WEIGHT_SHA256:
        raise ValueError("VQGAN tokenizer identity mismatch")
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "tokenizer_sha256": WEIGHT_SHA256,
        "preprocessing": protocol["preprocessing"],
        "test_accessed": False,
    }


def validate_shard(payload: dict, identity: dict, split: str, start: int, stop: int) -> None:
    if payload.get("identity") != identity or payload.get("split") != split:
        raise ValueError("Token shard identity mismatch")
    if payload.get("start") != start or payload.get("stop") != stop:
        raise ValueError("Token shard range mismatch")
    ids = payload.get("ids")
    labels = payload.get("labels")
    if not isinstance(ids, torch.Tensor) or ids.shape != (stop - start, 256) or ids.dtype != torch.int16:
        raise ValueError("Invalid token shard tensor")
    if not isinstance(labels, torch.Tensor) or labels.shape != (stop - start,):
        raise ValueError("Invalid token shard labels")
    if bool((ids < 0).any()) or bool((ids >= 1024).any()):
        raise ValueError("Token shard contains an invalid codebook ID")


def snapshot(protocol: dict, identity: dict, started: float, active_split: str | None = None) -> dict:
    split_rows = protocol["splits"]
    shard_size = int(protocol["shard_size"])
    cached_images = 0
    complete = True
    by_split = {}
    for split, rows in split_rows.items():
        split_dir = OUTPUT / split
        count = 0
        for index in range(expected_shards(int(rows), shard_size)):
            start = index * shard_size
            stop = min(start + shard_size, int(rows))
            path = split_dir / f"{index:05d}.pt"
            if path.exists():
                validate_shard(torch.load(path, map_location="cpu", weights_only=True), identity, split, start, stop)
                count += stop - start
        by_split[split] = {"cached_images": count, "planned_images": int(rows), "complete": count == int(rows)}
        cached_images += count
        complete = complete and count == int(rows)
    payload = {
        "complete": complete,
        "active_split": active_split,
        "cached_images": cached_images,
        "planned_images": sum(int(value) for value in split_rows.values()),
        "percent": round(100 * cached_images / sum(int(value) for value in split_rows.values()), 6),
        "splits": by_split,
        "elapsed_this_run_seconds": round(time.monotonic() - started, 3),
        "test_accessed": False,
    }
    atomic_json(OUTPUT / "progress.json", payload)
    return payload


@torch.inference_mode()
def cache(args: argparse.Namespace) -> int:
    started = time.monotonic()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    identity = validate_protocol(protocol)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUTPUT / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Existing Stage-21 cache has a different identity")
    atomic_json(identity_path, identity)
    torch.set_num_threads(args.threads)
    model = PretrainedVQGAN().to(args.device)
    shard_size = int(protocol["shard_size"])

    for split, planned in protocol["splits"].items():
        records = read_records(split)
        if len(records) != int(planned):
            raise ValueError(f"Unexpected {split} record count")
        split_dir = OUTPUT / split
        split_dir.mkdir(exist_ok=True)
        for index in range(expected_shards(len(records), shard_size)):
            start = index * shard_size
            stop = min(start + shard_size, len(records))
            shard_path = split_dir / f"{index:05d}.pt"
            if shard_path.exists():
                validate_shard(torch.load(shard_path, map_location="cpu", weights_only=True), identity, split, start, stop)
                continue
            images = []
            tokens = []
            for row in records[start:stop]:
                images.append(load_image(row["image"]))
                if len(images) == args.batch_size:
                    tokens.append(model.encode(torch.stack(images).to(args.device)).to(torch.int16).cpu())
                    images.clear()
            if images:
                tokens.append(model.encode(torch.stack(images).to(args.device)).to(torch.int16).cpu())
            ids = torch.cat(tokens)
            payload = {
                "identity": identity,
                "split": split,
                "start": start,
                "stop": stop,
                "ids": ids,
                "labels": torch.tensor([row["label"] for row in records[start:stop]], dtype=torch.int16),
            }
            validate_shard(payload, identity, split, start, stop)
            atomic_save(payload, shard_path)
            state = snapshot(protocol, identity, started, split)
            print(json.dumps(state), flush=True)
            if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds and not state["complete"]:
                return 75
    snapshot(protocol, identity, started)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0 or args.max_seconds < 0:
        parser.error("batch size/threads must be positive and max-seconds nonnegative")
    raise SystemExit(cache(args))


if __name__ == "__main__":
    main()
