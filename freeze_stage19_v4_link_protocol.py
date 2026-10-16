"""Freeze the Stage-19 V4 end-to-end communication evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE19_V4_LINK_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-19 protocol is already frozen")
    stage18 = json.loads((ROOT / "STAGE18_SUMMARY.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "STAGE18_AUDIT.json").read_text(encoding="utf-8"))
    if stage18.get("status") != "completed" or audit.get("status") != "passed":
        raise ValueError("Stage 18 is incomplete")
    if audit.get("test_accessed") or audit.get("stage14_holdout_used"):
        raise ValueError("Stage-18 isolation failed")
    checkpoint = ROOT / "runs/stage18_strong_cmi/best.pt"
    if sha256(checkpoint) != audit["best_checkpoint_sha256"]:
        raise ValueError("Stage-18 checkpoint identity changed")
    stage17 = json.loads((ROOT / "STAGE17_PERCEPTUAL_PROTOCOL.json").read_text(encoding="utf-8"))
    modes = stage17["modes"]
    conditions = [dict(condition) for condition in stage17["conditions"]]
    implementation = [
        "evaluate_stage19_v4_link.py",
        "tokcom_image_metrics.py",
        "pretrained_vq.py",
        "tokcom_arq.py",
        "context_maskgit_v3.py",
        "context_maskgit_v4.py",
        "digital_link.py",
    ]
    protocol = {
        "stage": 19,
        "status": "frozen",
        "purpose": "V4 stronger-CMI end-to-end digital-link, ARQ, VQGAN, CLIP and LPIPS evaluation",
        "evidence_level": "development evaluation; direct same-condition comparison with Stage 17; not a blind test",
        "implementation_sha256": {name: sha256(ROOT / name) for name in implementation},
        "checkpoint": "runs/stage18_strong_cmi/best.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "stage17_protocol_sha256": sha256(ROOT / "STAGE17_PERCEPTUAL_PROTOCOL.json"),
        "stage17_summary_sha256": sha256(ROOT / "STAGE17_PERCEPTUAL_SUMMARY.json"),
        "validation_cache": stage17["validation_cache"],
        "validation_cache_sha256": stage17["validation_cache_sha256"],
        "dataset_root": stage17["dataset_root"],
        "clip_checkpoint": stage17["clip_checkpoint"],
        "clip_checkpoint_sha256": stage17["clip_checkpoint_sha256"],
        "vqgan_checkpoint": stage17["vqgan_checkpoint"],
        "vqgan_checkpoint_sha256": stage17["vqgan_checkpoint_sha256"],
        "images": stage17["images"],
        "max_rounds": stage17["max_rounds"],
        "recovery_steps": stage17["recovery_steps"],
        "selected_stage16_policy": stage17["selected_stage16_policy"],
        "modes": modes,
        "conditions": conditions,
        "preview_indices": stage17["preview_indices"],
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUTPUT), "conditions": len(conditions), "sha256": sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
