"""Pre-register Stage-22 training before the full token cache is available."""
from __future__ import annotations

import json
from datetime import datetime

from train_stage22_contextual_tokcom import OUT, ROOT, sha256_file


OUTPUT = ROOT / "STAGE22_TRAINING_PLAN.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-22 training plan is already pre-registered")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Stage-22 output exists before training pre-registration")
    data = json.loads((ROOT / "data/stage20_food101/manifest.json").read_text(encoding="utf-8"))
    context = json.loads((ROOT / "STAGE21_CLIP_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
    cache_protocol = ROOT / "STAGE21_TOKEN_CACHE_PROTOCOL.json"
    initialization = ROOT / "runs/stage18_strong_cmi/best.pt"
    if data.get("test_accessed") or data.get("test_pixels_extracted") or context.get("test_accessed"):
        raise ValueError("Food-101 official test isolation failed before pre-registration")
    implementations = [
        "context_maskgit_v5.py",
        "context_maskgit_v4.py",
        "context_maskgit_v3.py",
        "models.py",
        "finalize_stage21_token_cache.py",
        "train_stage22_contextual_tokcom.py",
    ]
    payload = {
        "stage": 22,
        "status": "pre_registered",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "101-class CLIP-text-conditioned MaskGIT scale training",
        "evidence_level": "public Food-101 development training; not ImageNet100 and not a final test",
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in implementations},
        "stage21_token_cache_protocol_sha256": sha256_file(cache_protocol),
        "initialization": {
            "checkpoint": "runs/stage18_strong_cmi/best.pt",
            "checkpoint_sha256": sha256_file(initialization),
            "source_model": "Stage-18 ContextMaskGIT V4, 10-class Imagenette",
            "strategy": "copy token receiver and trained FiLM/prior; replace class lookup with zero-init CLIP text adapter around V4 null context",
        },
        "model": {
            "name": "ContextMaskGITV5",
            "classes": 101,
            "class_embeddings": "data/stage20_food101/clip_class_embeddings.pt",
            "class_embedding_dim": 512,
            "context_dropout": 0.15,
            "context_prior_weight": 0.15,
        },
        "data": {
            "dataset": "Food-101",
            "training_images": 70700,
            "development_images": 5050,
            "official_test_images_sealed": 25250,
            "label_bits": 7,
        },
        "training": {
            "seed": 20261022,
            "steps": 12000,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.00001,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "packet_tokens": 4,
            "per_choices": [0.1, 0.3, 0.5],
            "checkpoint_every": 250,
            "validation_every": 1000,
            "effective_sample_draws": 384000,
        },
        "validation": {
            "subset_seed": 2026102201,
            "images_per_class": 10,
            "images": 1010,
            "chunk_images_per_class": 1,
            "chunk_seed": 22100,
            "per": 0.3,
            "seeds": [22101, 22102, 22103],
            "batch_size": 32,
            "controls": ["correct class text", "wrong class text", "no class text"],
            "selection_rule": "minimum correct-context mean masked CE; ties use one-pass missing-token accuracy",
        },
        "paper_alignment": {
            "aligned": ["101 classes", "7-bit reliable class CMI", "256 VQ tokens", "1024-entry public MaskGIT VQGAN"],
            "approximate": ["Food-101 instead of ImageNet100", "CLIP class-name text instead of paper internal large-model cross-modal context", "trained receiver initialized from the engineering V4 approximation"],
        },
        "test_accessed": False,
        "test_pixels_extracted": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUTPUT), "sha256": sha256_file(OUTPUT), "steps": payload["training"]["steps"]}))


if __name__ == "__main__":
    main()
