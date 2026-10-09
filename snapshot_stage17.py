"""Write a compact Stage-17 progress snapshot."""
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage17_perceptual"


def main() -> None:
    protocol = json.loads((ROOT / "STAGE17_PERCEPTUAL_PROTOCOL.json").read_text(encoding="utf-8"))
    reference = len(list((RUN / "reference").glob("*.pt"))) if (RUN / "reference").exists() else 0
    completed = []
    active = None
    for condition in protocol["conditions"]:
        snr = str(float(condition["snr_esn0_db"])).replace(".", "p")
        key = f'{condition["channel"]}_{snr}db_seed{condition["seed"]}_{condition["mode"]}'
        directory = RUN / key
        if (directory / "summary.json").exists():
            completed.append(key)
        elif directory.exists() and active is None:
            active = {"key": key, "per_image_completed": len(list((directory / "per_image").glob("*.json")))}
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8")) if (RUN / "status.json").exists() else {}
    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_images": reference,
        "reference_planned": protocol["images"],
        "completed_conditions": len(completed),
        "planned_conditions": len(protocol["conditions"]),
        "active_condition": active,
        "completed": bool(status.get("completed")),
        "test_accessed": bool(status.get("test_accessed")),
        "stage14_holdout_used": bool(status.get("stage14_holdout_used")),
    }
    (ROOT / "STAGE17_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
