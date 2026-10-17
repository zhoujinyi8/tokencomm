"""Freeze the large Food-101 VQGAN token-cache protocol."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cache_stage21_food101_tokens import ROOT, sha256_file
from pretrained_vq import WEIGHT_SHA256


OUTPUT = ROOT / "STAGE21_TOKEN_CACHE_PROTOCOL.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-21 token-cache protocol is already frozen")
    context = json.loads((ROOT / "STAGE21_CLIP_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "data/stage20_food101/manifest.json").read_text(encoding="utf-8"))
    if context.get("status") != "prepared" or context.get("test_accessed"):
        raise ValueError("CLIP class context has not passed its isolation gate")
    if data.get("status") != "prepared" or data.get("test_accessed") or data.get("test_pixels_extracted"):
        raise ValueError("Food-101 development data is not prepared with sealed test pixels")
    inputs = [
        "data/stage20_food101/manifest.json",
        "data/stage20_food101/train.jsonl",
        "data/stage20_food101/dev.jsonl",
        "data/stage20_food101/clip_class_embeddings.pt",
        "STAGE21_CLIP_CONTEXT_MANIFEST.json",
        "cache_stage21_food101_tokens.py",
        "pretrained_vq.py",
    ]
    payload = {
        "stage": 21,
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "VQGAN token cache for 101-class ContextMaskGIT V5 scale training",
        "input_sha256": {name: sha256_file(ROOT / name) for name in inputs},
        "tokenizer": "public MaskGIT VQGAN f16 256, 1024 entries",
        "tokenizer_sha256": WEIGHT_SHA256,
        "preprocessing": "PIL ImageOps.fit 256x256 LANCZOS RGB float32 [0,1]",
        "splits": {"train": int(data["train_images"]), "dev": int(data["dev_images"])},
        "shard_size": 32,
        "official_test_images": int(data["sealed_test_images"]),
        "official_test_policy": "pixels remain compressed and excluded from caching, training, and selection",
        "test_accessed": False,
        "test_pixels_extracted": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUTPUT), "sha256": sha256_file(OUTPUT), "images": sum(payload["splits"].values())}))


if __name__ == "__main__":
    main()
