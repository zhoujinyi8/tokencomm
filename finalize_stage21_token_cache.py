"""Audit Stage-21 shards and build train/dev aggregate tensors."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cache_stage21_food101_tokens import (
    DATA,
    OUTPUT,
    PROTOCOL_PATH,
    atomic_json,
    atomic_save,
    expected_shards,
    read_records,
    sha256_file,
    validate_protocol,
    validate_shard,
)


SUMMARY = OUTPUT / "summary.json"


def aggregate_split(protocol: dict, identity: dict, split: str) -> dict:
    total = int(protocol["splits"][split])
    shard_size = int(protocol["shard_size"])
    ids, labels = [], []
    for index in range(expected_shards(total, shard_size)):
        start = index * shard_size
        stop = min(start + shard_size, total)
        path = OUTPUT / split / f"{index:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage-21 shard: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        validate_shard(shard, identity, split, start, stop)
        ids.append(shard["ids"])
        labels.append(shard["labels"])
    ids_tensor = torch.cat(ids)
    labels_tensor = torch.cat(labels)
    records = read_records(split)
    record_labels = torch.tensor([row["label"] for row in records], dtype=torch.int16)
    if len(records) != total or not torch.equal(labels_tensor, record_labels):
        raise ValueError(f"{split} aggregate labels do not align with JSONL records")
    path = OUTPUT / f"{split}.pt"
    atomic_save(
        {
            "ids": ids_tensor,
            "labels": labels_tensor,
            "identity": identity,
            "split": split,
            "records_sha256": sha256_file(DATA / f"{split}.jsonl"),
        },
        path,
    )
    return {
        "path": str(path.relative_to(Path(__file__).resolve().parent)).replace("\\", "/"),
        "images": total,
        "shape": list(ids_tensor.shape),
        "dtype": str(ids_tensor.dtype),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def finalize() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    identity = validate_protocol(protocol)
    progress_path = OUTPUT / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    if not progress.get("complete") or progress.get("test_accessed"):
        raise RuntimeError("Stage-21 shards are incomplete or violate test isolation")
    if SUMMARY.exists():
        existing = json.loads(SUMMARY.read_text(encoding="utf-8"))
        valid = existing.get("protocol_sha256") == identity["protocol_sha256"]
        for split in protocol["splits"]:
            item = existing.get("aggregates", {}).get(split, {})
            path = Path(__file__).resolve().parent / item.get("path", "missing")
            valid = valid and path.is_file() and sha256_file(path) == item.get("sha256")
        if valid:
            return existing
    aggregates = {split: aggregate_split(protocol, identity, split) for split in protocol["splits"]}
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "completed": True,
        "protocol_sha256": identity["protocol_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "classes": len(manifest["classes"]),
        "class_names": manifest["classes"],
        "aggregates": aggregates,
        "test_pixels_extracted": False,
        "test_accessed": False,
    }
    atomic_json(SUMMARY, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(finalize()))
