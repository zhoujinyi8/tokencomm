"""Write a compact Stage-18 training progress snapshot."""
import json
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage18_strong_cmi"


def main() -> None:
    protocol = json.loads((ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json").read_text(encoding="utf-8"))
    recovery = torch.load(RUN / "recovery.pt", weights_only=True) if (RUN / "recovery.pt").exists() else {}
    rows = [json.loads(line) for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines()] if (RUN / "history.jsonl").exists() else []
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8")) if (RUN / "status.json").exists() else {}
    validations = [row for row in rows if "validation" in row]
    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "step": int(recovery.get("step", 0)),
        "planned_steps": int(protocol["training"]["steps"]),
        "history_rows": len(rows),
        "best_step": int(recovery.get("best_step", 0)),
        "latest_validation": validations[-1]["validation"] if validations else None,
        "completed": bool(status.get("completed")),
        "test_accessed": bool(status.get("test_accessed")),
        "stage14_holdout_used": bool(status.get("stage14_holdout_used")),
    }
    (ROOT / "STAGE18_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
