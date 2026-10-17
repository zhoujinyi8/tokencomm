"""Bind the pre-registered Stage-23 link plan to audited Stage-22 artifacts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from train_stage22_contextual_tokcom import sha256_file


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE23_V5_LINK_PROTOCOL.json"


def main() -> None:
    if OUTPUT.exists():
        print(json.dumps({"already_frozen": True, "sha256": sha256_file(OUTPUT)}))
        return
    plan_path = ROOT / "STAGE23_V5_LINK_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary_path = ROOT / "STAGE22_SUMMARY.json"
    audit_path = ROOT / "STAGE22_AUDIT.json"
    checkpoint = ROOT / "runs/stage22_contextual_tokcom/best.pt"
    cache = ROOT / "data/stage20_food101/token_cache_stage21/dev.pt"
    records = ROOT / "data/stage20_food101/dev.jsonl"
    for path in (summary_path, audit_path, checkpoint, cache, records):
        if not path.exists():
            raise FileNotFoundError(f"Stage-23 prerequisite is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if plan.get("status") != "pre_registered" or plan.get("test_accessed") or plan.get("test_pixels_extracted"):
        raise ValueError("Stage-23 plan was not safely pre-registered")
    if summary.get("status") != "completed" or summary.get("audit", {}).get("status") != "passed":
        raise ValueError("Stage-22 summary did not pass")
    if audit.get("status") != "passed" or audit.get("test_accessed") or audit.get("test_pixels_extracted"):
        raise ValueError("Stage-22 audit did not preserve test isolation")
    if sha256_file(ROOT / "STAGE22_TRAINING_PLAN.json") != plan["stage22_training_plan_sha256"]:
        raise ValueError("Stage-22 plan changed after Stage-23 pre-registration")
    link = plan["link"]
    calibration_modes = ["semantic_only", "full_arq"] + [f"selective_{value:g}" for value in link["threshold_candidates"]]
    calibration_conditions = [
        {**channel, "seed": seed, "mode": mode}
        for channel in link["channels"] for seed in link["seeds"] for mode in calibration_modes
    ]
    evaluation_conditions = [
        {**channel, "seed": seed, "mode": mode}
        for channel in link["channels"] for seed in link["seeds"] for mode in link["evaluation_modes"]
    ]
    implementation = [
        "evaluate_stage23_v5_link.py", "finalize_stage23.py", "stage23_link_common.py", "tokcom_arq.py",
        "context_maskgit_v5.py", "context_maskgit_v4.py", "context_maskgit_v3.py", "digital_link.py",
    ]
    payload = dict(plan)
    payload.update({
        "status": "frozen",
        "protocol_frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pre_registered_plan_sha256": sha256_file(plan_path),
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in implementation},
        "stage22_summary_sha256": sha256_file(summary_path),
        "stage22_audit_sha256": sha256_file(audit_path),
        "checkpoint": "runs/stage22_contextual_tokcom/best.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "development_cache": "data/stage20_food101/token_cache_stage21/dev.pt",
        "development_cache_sha256": sha256_file(cache),
        "development_records": "data/stage20_food101/dev.jsonl",
        "development_records_sha256": sha256_file(records),
        "calibration_conditions": calibration_conditions,
        "evaluation_conditions": evaluation_conditions,
    })
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "calibration_conditions": len(calibration_conditions), "evaluation_conditions": len(evaluation_conditions), "sha256": sha256_file(OUTPUT)}))


if __name__ == "__main__":
    main()
