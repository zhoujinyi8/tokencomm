"""Freeze Stage-15 contextual TokCom development training inputs and code."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE15_CONTEXTUAL_TOKCOM_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    executable = shutil.which("git")
    if executable is None:
        candidates = [
            Path("C:/Program Files/Git/cmd/git.exe"),
            Path("C:/Program Files/Git/bin/git.exe"),
        ]
        executable = next((str(path) for path in candidates if path.exists()), None)
    if executable is None:
        raise FileNotFoundError("git.exe is required to bind the implementation commit")
    return subprocess.run(
        [executable, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    if OUTPUT.exists():
        print(json.dumps({"already_frozen": True, "sha256": sha256(OUTPUT)}))
        return
    stage14 = json.loads((ROOT / "runs/stage14_tokcom_final/status.json").read_text(encoding="utf-8"))
    if not stage14.get("completed") or not stage14.get("test_accessed") or stage14.get("used_for_training"):
        raise ValueError("Stage-14 completion/provenance audit failed")
    base = ROOT / "runs/stage13_tokcom_large/best.pt"
    checkpoint = torch.load(base, weights_only=True)
    if checkpoint.get("model") != "context_maskgit_v2":
        raise ValueError("Stage-15 requires the frozen V2 initialization checkpoint")
    train_cache = torch.load(ROOT / "data/imagenette_stage10/token_cache/train.pt", weights_only=True)
    validation_cache = torch.load(ROOT / "data/imagenette_stage10/token_cache/val.pt", weights_only=True)
    if train_cache["identity"] != validation_cache["identity"]:
        raise ValueError("Training/validation cache identity mismatch")
    implementation = [
        "context_maskgit_v3.py",
        "tokcom_arq.py",
        "train_stage15_contextual_tokcom.py",
        "context_maskgit_v2.py",
        "models.py",
        "digital_link.py",
    ]
    data_files = [
        "data/imagenette_stage10/manifest.json",
        "data/imagenette_stage10/train.jsonl",
        "data/imagenette_stage10/val.jsonl",
        "data/imagenette_stage10/token_cache/train.pt",
        "data/imagenette_stage10/token_cache/val.pt",
    ]
    payload = {
        "stage": 15,
        "purpose": "Train a cross-modal class-conditioned MaskGIT receiver and implement TokCom ARQ/TCE",
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_commit": git_head(),
        "implementation_sha256": {name: sha256(ROOT / name) for name in implementation},
        "data": {
            "training_images": 8000,
            "validation_images": 150,
            "hashes": {name: sha256(ROOT / name) for name in data_files},
            "selection_split": "data/imagenette_stage10/val.jsonl",
            "stage14_holdout_used": False,
        },
        "tokenizer_sha256": train_cache["identity"]["tokenizer_sha256"],
        "initialization": {
            "checkpoint": "runs/stage13_tokcom_large/best.pt",
            "checkpoint_sha256": sha256(base),
            "source_model": "ContextMaskGITv2 at Stage-13 step 6000",
            "strategy": "copy every V2 weight; add a zero no-context embedding and small class embeddings",
        },
        "context": {
            "source": "Imagenette ten-class label",
            "classes": 10,
            "minimum_label_bits": 4,
            "reliability": "perfect receiver recovery assumed, matching the paper case-study assumption",
            "paper_difference": "The paper uses a 7-bit ImageNet100 label; this dataset has ten classes.",
        },
        "model": {
            "name": "ContextMaskGITv3",
            "width": int(checkpoint["config"]["d"]),
            "transformer_layers": int(checkpoint["config"]["layers"]),
            "local_blocks": int(checkpoint["config"]["local_blocks"]),
            "classes": 10,
            "context_dropout": 0.15,
            "classifier_free_guidance": True,
        },
        "training": {
            "seed": 20261009,
            "steps": 2000,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "minimum_learning_rate": 0.00002,
            "weight_decay": 0.01,
            "packet_tokens": 4,
            "per_choices": [0.1, 0.3, 0.5],
            "checkpoint_every": 100,
            "validation_every": 250,
        },
        "validation": {
            "per": 0.3,
            "seeds": [15101, 15102, 15103],
            "recovery_steps": 8,
            "guidance_scale": 1.0,
            "guidance_sweep": [0.0, 0.5, 1.0, 1.5, 2.0],
            "controls": ["correct context", "wrong context", "no context"],
            "selection_rule": "minimum correct-context mean masked-token cross entropy; ties use exact recovery",
        },
        "arq": {
            "analytical": "T=1/(1-PER); TCE=H*W/(T*N*log2(Q))",
            "policies": ["all_failed", "confidence"],
            "confidence_semantics": "retransmit CRC-failed packets whose mean predicted token likelihood is below threshold",
        },
        "test_accessed_for_training_or_selection": False,
        "stage14_holdout_used_for_training_or_selection": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "sha256": sha256(OUTPUT), "steps": payload["training"]["steps"]}))


if __name__ == "__main__":
    main()
