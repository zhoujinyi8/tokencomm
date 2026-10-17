"""Consolidate the five-hour TokCom/TaiChi reproduction state."""
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    stage19 = read(ROOT / "STAGE19_PROGRESS.json")
    stage19_audit = read(ROOT / "STAGE19_AUDIT.json")
    stage20 = read(ROOT / "STAGE20_PROGRESS.json")
    stage20_manifest = read(ROOT / "data/stage20_food101/manifest.json")
    stage21_embeddings = ROOT / "data/stage20_food101/clip_class_embeddings.pt"
    stage21_cache = read(ROOT / "data/stage20_food101/token_cache_stage21/progress.json")
    stage21_summary = read(ROOT / "data/stage20_food101/token_cache_stage21/summary.json")
    stage22_plan = read(ROOT / "STAGE22_TRAINING_PLAN.json")
    stage22_protocol = read(ROOT / "STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json")
    stage22_status = read(ROOT / "runs/stage22_contextual_tokcom/status.json")
    stage22_audit = read(ROOT / "STAGE22_AUDIT.json")
    stage22_recovery = ROOT / "runs/stage22_contextual_tokcom/recovery.pt"
    stage23_plan = read(ROOT / "STAGE23_V5_LINK_PLAN.json")
    stage23_protocol = read(ROOT / "STAGE23_V5_LINK_PROTOCOL.json")
    stage23_status = read(ROOT / "runs/stage23_v5_link/status.json")
    stage23_audit = read(ROOT / "STAGE23_AUDIT.json")
    if not stage19.get("completed"):
        next_action = "continue Stage-19 frozen V4 link evaluation"
    elif stage19_audit.get("status") != "passed":
        next_action = "run Stage-19 fail-closed final audit"
    elif not stage20.get("prepared"):
        next_action = "continue Stage-20 public data download/preparation"
    elif not stage21_embeddings.exists():
        next_action = "build pinned CLIP class-name embeddings for ContextMaskGIT V5"
    elif not stage21_cache.get("complete"):
        next_action = "continue resumable Food-101 VQGAN token caching"
    elif not stage21_summary.get("completed"):
        next_action = "audit and aggregate Stage-21 VQGAN token shards"
    elif stage22_protocol.get("status") != "frozen":
        next_action = "freeze 101-class ContextMaskGIT V5 training protocol"
    elif not stage22_status.get("completed"):
        next_action = "continue resumable 12,000-step ContextMaskGIT V5 training"
    elif stage22_audit.get("status") != "passed":
        next_action = "run fail-closed Stage-22 audit and figures"
    elif stage23_protocol.get("status") != "frozen":
        next_action = "freeze pre-registered V5 link calibration protocol"
    elif not stage23_status.get("completed"):
        next_action = "continue V5 confidence-threshold calibration and link evaluation"
    elif stage23_audit.get("status") != "passed":
        next_action = "run fail-closed Stage-23 link audit and figures"
    else:
        next_action = "Stage-23 V5 development link evaluation completed"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cadence_hours": 5,
        "active_minutes": 20,
        "stage19_completed_conditions": stage19.get("completed_conditions", 0),
        "stage19_planned_conditions": stage19.get("planned_conditions", 24),
        "stage19_audit": stage19_audit.get("status", "pending"),
        "stage20_download_percent": stage20.get("download_percent", 0.0),
        "stage20_prepared": stage20.get("prepared", False),
        "stage20_train_records": stage20.get("train_records", 0),
        "stage20_dev_records": stage20.get("dev_records", 0),
        "stage20_test_pixels_extracted": stage20.get("test_pixels_extracted", False),
        "stage20_test_accessed": stage20.get("test_accessed", False),
        "stage20_archive_sha256": stage20_manifest.get("archive_sha256"),
        "v5_implementation_present": (ROOT / "context_maskgit_v5.py").exists(),
        "v5_class_embeddings_present": stage21_embeddings.exists(),
        "stage21_cached_images": stage21_cache.get("cached_images", 0),
        "stage21_planned_images": stage21_cache.get("planned_images", 75750),
        "stage21_cache_percent": stage21_cache.get("percent", 0.0),
        "stage21_aggregated": stage21_summary.get("completed", False),
        "stage22_plan_preregistered": stage22_plan.get("status") == "pre_registered",
        "stage22_protocol_frozen": stage22_protocol.get("status") == "frozen",
        "stage22_training_completed": stage22_status.get("completed", False),
        "stage22_audit": stage22_audit.get("status", "pending"),
        "stage22_recovery_present": stage22_recovery.exists(),
        "stage23_plan_preregistered": stage23_plan.get("status") == "pre_registered",
        "stage23_protocol_frozen": stage23_protocol.get("status") == "frozen",
        "stage23_link_completed": stage23_status.get("completed", False),
        "stage23_audit": stage23_audit.get("status", "pending"),
        "next_action": next_action,
    }
    (ROOT / "GOAL_5H_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
