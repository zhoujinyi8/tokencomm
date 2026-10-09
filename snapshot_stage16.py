"""Write a compact Stage-16 ARQ/TCE progress snapshot."""
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage16_arq_tce"


def main() -> None:
    rows = json.loads((RUN / "results.json").read_text(encoding="utf-8")) if (RUN / "results.json").exists() else []
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8")) if (RUN / "status.json").exists() else None
    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_conditions": len(rows),
        "planned_conditions": 30,
        "latest_condition": (
            {key: rows[-1][key] for key in ["channel", "snr_esn0_db", "seed", "policy_name"]}
            if rows else None
        ),
        "completed": bool(status and status.get("completed")),
        "test_accessed": bool(status and status.get("test_accessed")),
        "stage14_holdout_used": bool(status and status.get("stage14_holdout_used")),
    }
    (ROOT / "STAGE16_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
