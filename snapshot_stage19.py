"""Write a compact Stage-19 progress snapshot with live paired V3/V4 deltas."""
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage19_v4_link"
BASELINE = ROOT / "runs/stage17_perceptual"
FIELDS = ("token_accuracy", "psnr", "clip_cosine", "lpips", "vqgan_baseline_psnr")


def paired_deltas(baseline: Path, candidate: Path) -> dict | None:
    baseline_rows = baseline / "per_image"
    candidate_rows = candidate / "per_image"
    names = sorted({path.name for path in baseline_rows.glob("*.json")} & {path.name for path in candidate_rows.glob("*.json")})
    if not names:
        return None
    old = [json.loads((baseline_rows / name).read_text(encoding="utf-8")) for name in names]
    new = [json.loads((candidate_rows / name).read_text(encoding="utf-8")) for name in names]
    if any(left["index"] != right["index"] or left["sha256"] != right["sha256"] for left, right in zip(old, new)):
        raise ValueError("Stage-19 live comparison is not image-aligned with Stage 17")
    return {
        "paired_images": len(names),
        **{
            f"{field}_delta_mean": fmean(float(right[field]) - float(left[field]) for left, right in zip(old, new))
            for field in FIELDS
        },
    }


def main() -> None:
    protocol = json.loads((ROOT / "STAGE19_V4_LINK_PROTOCOL.json").read_text(encoding="utf-8"))
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
            active = {
                "key": key,
                "per_image_completed": len(list((directory / "per_image").glob("*.json"))),
                "stage17_to_stage19_live_delta": paired_deltas(BASELINE / key, directory),
            }
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
    (ROOT / "STAGE19_PROGRESS.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()

