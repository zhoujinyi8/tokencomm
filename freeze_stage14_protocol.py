"""Freeze the Stage-14 TokCom blind-test protocol after development is complete."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "STAGE14_TOKCOM_FINAL_PROTOCOL.json"
SOURCE = ROOT / "work/stage12_source/validation-00000-of-00001.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditions():
    return [
        {"channel": channel, "snr_esn0_db": float(snr), "seed": seed}
        for channel in ["awgn", "rayleigh"]
        for snr in [0, 5, 10, 20]
        for seed in [14201, 14202, 14203]
    ]


def main() -> None:
    if PROTOCOL.exists():
        print(json.dumps({"already_frozen": True, "sha256": sha256(PROTOCOL)}))
        return
    required = [
        "STAGE14_HOLDOUT_PLAN.md",
        "STAGE13_TOKCOM_REPORT.md", "STAGE13_AUDIT.json", "STAGE13_PHYSICAL_REPORT.md",
        "TOKCOM_TAICHI_FAIR_COMPARISON.md", "TOKCOM_TAICHI_FAIR_COMPARISON.json",
        "STAGE13_PHYSICAL_DEV_PROTOCOL.json", "runs/stage13_physical_dev/status.json",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Stage-13 development is not fully reported: {missing}")
    audit = json.loads((ROOT / "STAGE13_AUDIT.json").read_text(encoding="utf-8"))
    physical_status = json.loads((ROOT / "runs/stage13_physical_dev/status.json").read_text(encoding="utf-8"))
    physical_protocol = json.loads((ROOT / "STAGE13_PHYSICAL_DEV_PROTOCOL.json").read_text(encoding="utf-8"))
    if not audit.get("audit_passed") or audit.get("test_accessed") or audit.get("used_for_training"):
        raise ValueError("Stage-13 training audit is incomplete or contaminated")
    if not physical_status.get("completed") or physical_status.get("test_accessed"):
        raise ValueError("Stage-13 physical development sweep is incomplete or contaminated")
    checkpoint = ROOT / physical_protocol["checkpoint"]
    if sha256(checkpoint) != physical_protocol["checkpoint_sha256"]:
        raise ValueError("Selected Stage-13 checkpoint hash changed")
    stage12 = json.loads((ROOT / "STAGE12_FINAL_TEST_PROTOCOL.json").read_text(encoding="utf-8"))
    if sha256(SOURCE) != stage12["source_validation_sha256"]:
        raise ValueError("Public Imagenette validation container hash changed")
    exclusions = list(stage12["exclusion_manifests"])
    stage12_test = ROOT / "data/stage12_final_holdout/test.jsonl"
    exclusions.append({"path": stage12_test.relative_to(ROOT).as_posix(), "sha256": sha256(stage12_test)})
    scripts = [
        "freeze_stage14_protocol.py", "prepare_stage14_holdout.py", "evaluate_stage14_final.py",
        "report_stage14_final.py", "digital_link.py", "context_maskgit_v2.py", "cache_real_tokens.py",
    ]
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "protocol": "Stage-14 preregistered TokCom blind test on a new disjoint Imagenette holdout",
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_commit": git_head,
        "development_protocol_sha256": sha256(ROOT / "STAGE13_TOKCOM_PROTOCOL.json"),
        "holdout_plan_sha256": sha256(ROOT / "STAGE14_HOLDOUT_PLAN.md"),
        "development_audit_sha256": sha256(ROOT / "STAGE13_AUDIT.json"),
        "physical_development_protocol_sha256": sha256(ROOT / "STAGE13_PHYSICAL_DEV_PROTOCOL.json"),
        "selection_rule": "Stage-13 frozen minimum development cross-entropy rule; no post-test model selection",
        "checkpoint": physical_protocol["checkpoint"],
        "checkpoint_sha256": physical_protocol["checkpoint_sha256"],
        "tokenizer_sha256": physical_protocol["tokenizer_sha256"],
        "source": "ShaomuTan/imagenette 320px validation parquet",
        "source_validation_sha256": stage12["source_validation_sha256"],
        "selection_seed": 20261008,
        "per_class": 30,
        "expected_test_images": 300,
        "class_names": stage12["class_names"],
        "exclusion_manifests": exclusions,
        "holdout_directory": "data/stage14_tokcom_final_holdout",
        "token_cache": "data/stage14_tokcom_final_holdout/token_cache/test.pt",
        "conditions": conditions(),
        "recovery_steps": 8,
        "evaluation_batch_size": 24,
        "link": physical_protocol["link"],
        "metrics": [
            "measured_per", "payload_ber", "recovered_all_token_accuracy",
            "maskgit_missing_accuracy", "spatial_missing_accuracy", "per_class_all_token_accuracy",
            "undetected_wrong_tokens", "seed_mean", "seed_population_std",
        ],
        "implementation_sha256": {name: sha256(ROOT / name) for name in scripts},
        "candidate_source_semantics_accessed_at_freeze": False,
        "source_container_hash_verified_at_freeze": True,
        "test_accessed_at_freeze": False,
        "used_for_training": False,
        "reporting_rule": "Evaluate and report every frozen condition; do not tune, retrain, replace the checkpoint, alter the holdout, or omit weak conditions after holdout materialization.",
        "pretraining_overlap": "The public VQGAN was pretrained on ImageNet and Imagenette is an ImageNet subset; this is a held-out downstream split, not an unseen-pretraining claim.",
    }
    PROTOCOL.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "conditions": len(payload["conditions"]), "sha256": sha256(PROTOCOL)}))


if __name__ == "__main__":
    main()
