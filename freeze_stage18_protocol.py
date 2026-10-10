"""Freeze stronger CMI training only after Stage 17 passes its audit."""
import json
from datetime import datetime
from pathlib import Path

from train_stage15_contextual_tokcom import sha256


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-18 protocol is already frozen")
    audit = json.loads((ROOT / "STAGE17_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("test_accessed") or audit.get("stage14_holdout_used"):
        raise ValueError("Stage-17 completion/isolation gate failed")
    initialization = ROOT / "runs/stage15_contextual_tokcom/best.pt"
    implementation = [
        "context_maskgit_v4.py",
        "train_stage18_contextual_tokcom.py",
        "context_maskgit_v3.py",
        "context_maskgit_v2.py",
        "models.py",
    ]
    data_files = [
        "data/imagenette_stage10/manifest.json",
        "data/imagenette_stage10/train.jsonl",
        "data/imagenette_stage10/val.jsonl",
        "data/imagenette_stage10/token_cache/train.pt",
        "data/imagenette_stage10/token_cache/val.pt",
    ]
    protocol = {
        "stage": 18,
        "purpose": "Strengthen the paper-case-study class CMI path after the Stage-17 weak-CMI finding",
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_level": "development training and model selection; final holdouts remain excluded",
        "implementation_sha256": {name: sha256(ROOT / name) for name in implementation},
        "data": {
            "training_images": 8000,
            "validation_images": 150,
            "hashes": {name: sha256(ROOT / name) for name in data_files},
            "stage14_holdout_used": False,
        },
        "tokenizer_sha256": "424e7f2b11d1d63691ae3fbab74cf6c417ecbe0d8f86187b71f3538c7b56f6d2",
        "initialization": {
            "checkpoint": "runs/stage15_contextual_tokcom/best.pt",
            "checkpoint_sha256": sha256(initialization),
            "source_model": "ContextMaskGITv3 at Stage-15 best step",
            "strategy": "exact V3 warm start; zero-init output FiLM and class-code prior",
        },
        "paper_alignment": {
            "case_study_cmi": "perfectly recovered image class label over a separate channel",
            "current_label_bits": 4,
            "paper_label_bits": 7,
            "difference": "Imagenette has 10 classes while the paper uses ImageNet100; the class-prior auxiliary objective is an engineering approximation not disclosed by the paper",
        },
        "model": {
            "name": "ContextMaskGITv4",
            "classes": 10,
            "context_dropout": 0.15,
            "context_prior_weight": 0.15,
            "new_condition_paths": ["output FiLM", "class-specific 1024-code prior"],
        },
        "training": {
            "seed": 20261010,
            "steps": 3000,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.00001,
            "weight_decay": 0.01,
            "packet_tokens": 4,
            "per_choices": [0.1, 0.3, 0.5],
            "checkpoint_every": 100,
            "validation_every": 250,
        },
        "validation": {
            "per": 0.3,
            "seeds": [18101, 18102, 18103],
            "recovery_steps": 8,
            "guidance_scale": 1.0,
            "guidance_sweep": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
            "controls": ["correct class", "wrong class", "no class"],
            "selection_rule": "minimum correct-class mean masked-token cross entropy; ties use exact recovery",
        },
        "test_accessed_for_training_or_selection": False,
        "stage14_holdout_used_for_training_or_selection": False,
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUTPUT), "sha256": sha256(OUTPUT), "steps": 3000}))


if __name__ == "__main__":
    main()
