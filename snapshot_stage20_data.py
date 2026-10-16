"""Write a compact Stage-20 download/preparation and isolation snapshot."""
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/stage20_food101"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main() -> None:
    plan = json.loads((ROOT / "STAGE20_FOOD101_SCALE_PLAN.json").read_text(encoding="utf-8"))
    part = DATA / "food-101.tar.gz.part"
    archive = DATA / "food-101.tar.gz"
    downloaded = archive.stat().st_size if archive.exists() else part.stat().st_size if part.exists() else 0
    expected = int(plan["source"]["content_length"])
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8")) if (DATA / "manifest.json").exists() else {}
    sealed = json.loads((DATA / "sealed_test_manifest.json").read_text(encoding="utf-8")) if (DATA / "sealed_test_manifest.json").exists() else {}
    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "downloaded_bytes": downloaded,
        "expected_bytes": expected,
        "download_percent": 100.0 * downloaded / expected,
        "archive_complete": archive.exists() and downloaded == expected,
        "archive_identity_recorded": (DATA / "archive_identity.json").exists(),
        "extracted_training_images": len(list((DATA / "extracted/food-101/images").glob("*/*.jpg"))),
        "train_records": count_lines(DATA / "train.jsonl"),
        "dev_records": count_lines(DATA / "dev.jsonl"),
        "prepared": manifest.get("status") == "prepared",
        "test_pixels_extracted": bool(manifest.get("test_pixels_extracted", sealed.get("pixels_extracted", False))),
        "test_accessed": bool(manifest.get("test_accessed", False)),
        "stage14_holdout_used": False,
    }
    (ROOT / "STAGE20_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
