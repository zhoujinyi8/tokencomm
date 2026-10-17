"""Freeze 101-class ContextMaskGIT V5 training after token-cache audit."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from train_stage22_contextual_tokcom import ROOT, sha256_file


OUTPUT = ROOT / "STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-22 protocol is already frozen")
    summary_path = ROOT / "data/stage20_food101/token_cache_stage21/summary.json"
    if not summary_path.exists():
        raise RuntimeError("Stage-21 token cache is not yet aggregated")
    cache = json.loads(summary_path.read_text(encoding="utf-8"))
    context = json.loads((ROOT / "STAGE21_CLIP_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    plan_path = ROOT / "STAGE22_TRAINING_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not cache.get("completed") or cache.get("test_accessed"):
        raise ValueError("Stage-21 token-cache audit has not passed")
    if context.get("status") != "prepared" or context.get("test_accessed"):
        raise ValueError("Stage-21 CLIP context has not passed")
    if plan.get("status") != "pre_registered" or plan.get("test_accessed") or plan.get("test_pixels_extracted"):
        raise ValueError("Stage-22 training plan was not safely pre-registered")
    for name, expected in plan["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Pre-registered Stage-22 implementation changed: {name}")
    if sha256_file(ROOT / "STAGE21_TOKEN_CACHE_PROTOCOL.json") != plan["stage21_token_cache_protocol_sha256"]:
        raise ValueError("Stage-21 token-cache protocol changed after Stage-22 pre-registration")
    initialization = ROOT / plan["initialization"]["checkpoint"]
    if sha256_file(initialization) != plan["initialization"]["checkpoint_sha256"]:
        raise ValueError("Stage-22 initialization changed after pre-registration")
    inputs = [
        "data/stage20_food101/token_cache_stage21/train.pt",
        "data/stage20_food101/token_cache_stage21/dev.pt",
        "data/stage20_food101/token_cache_stage21/summary.json",
        "data/stage20_food101/dev.jsonl",
        "data/stage20_food101/clip_class_embeddings.pt",
        "STAGE21_CLIP_CONTEXT_MANIFEST.json",
        "STAGE21_TOKEN_CACHE_PROTOCOL.json",
        "context_maskgit_v5.py",
        "context_maskgit_v4.py",
        "context_maskgit_v3.py",
        "models.py",
        "train_stage22_contextual_tokcom.py",
        "STAGE22_TRAINING_PLAN.json",
    ]
    payload = dict(plan)
    payload.update({
        "status": "frozen",
        "protocol_frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pre_registered_plan_sha256": sha256_file(plan_path),
        "input_sha256": {name: sha256_file(ROOT / name) for name in inputs},
        "tokenizer_sha256": cache["tokenizer_sha256"],
    })
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUTPUT), "sha256": sha256_file(OUTPUT), "steps": 12000}))


if __name__ == "__main__":
    main()
