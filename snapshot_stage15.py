"""Write a compact machine-readable Stage-15 progress snapshot."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage15_contextual_tokcom"


def main() -> None:
    rows = []
    history = RUN / "history.jsonl"
    if history.exists():
        rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
    validations = [row for row in rows if "validation" in row]
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8")) if (RUN / "status.json").exists() else None
    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_steps": int(rows[-1]["step"]) if rows else 0,
        "planned_steps": 2000,
        "validation_count": len(validations),
        "latest_validation": validations[-1]["validation"] if validations else None,
        "completed": bool(status and status.get("completed")),
        "test_accessed": bool(status and status.get("test_accessed")),
        "stage14_holdout_used": bool(status and status.get("stage14_holdout_used")),
    }
    (ROOT / "STAGE15_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
