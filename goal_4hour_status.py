"""Consolidate the four-hour TokCom/TaiChi reproduction state."""
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
    else:
        next_action = "freeze and launch Stage-21 V5 token caching/training"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cadence_hours": 4,
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
        "next_action": next_action,
    }
    (ROOT / "GOAL_4H_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
