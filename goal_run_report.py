"""Create an auditable before/after report for each five-hour worker run."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "GOAL_5H_RUN_ACTIVE.json"
LATEST_JSON = ROOT / "GOAL_5H_LATEST_RUN_REPORT.json"
LATEST_MD = ROOT / "GOAL_5H_LATEST_RUN_REPORT.md"
HISTORY = ROOT / "GOAL_5H_RUN_HISTORY.jsonl"


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def json_rows(path: Path) -> int:
    value = read_json(path, [])
    return len(value) if isinstance(value, list) else 0


def last_history_step(path: Path) -> int:
    if not path.exists():
        return 0
    last = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    return int(last["step"]) if last else 0


def snapshot(root: Path = ROOT) -> dict:
    cache = read_json(root / "data/stage20_food101/token_cache_stage21/progress.json")
    stage22 = read_json(root / "runs/stage22_contextual_tokcom/status.json")
    stage23 = read_json(root / "runs/stage23_v5_link/status.json")
    manifest = read_json(root / "data/stage20_food101/manifest.json")
    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage21_cached_images": int(cache.get("cached_images", 0)),
        "stage21_planned_images": int(cache.get("planned_images", 75750)),
        "stage21_complete": bool(cache.get("complete", False)),
        "stage21_process_elapsed_seconds": float(cache.get("elapsed_this_run_seconds", 0.0)),
        "stage22_step": last_history_step(root / "runs/stage22_contextual_tokcom/history.jsonl"),
        "stage22_complete": bool(stage22.get("completed", False)),
        "stage23_calibration_conditions": json_rows(root / "runs/stage23_v5_link/calibration_results.json"),
        "stage23_evaluation_conditions": json_rows(root / "runs/stage23_v5_link/evaluation_results.json"),
        "stage23_complete": bool(stage23.get("completed", False)),
        "test_pixels_extracted": bool(manifest.get("test_pixels_extracted", False)),
        "test_accessed": bool(manifest.get("test_accessed", False) or cache.get("test_accessed", False)),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def start() -> dict:
    if ACTIVE.exists():
        finish(
            "interrupted",
            1,
            message="Previous worker ended without reaching its completion reporter; Task Scheduler or the host likely stopped the process externally.",
        )
    payload = {"run_started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "before": snapshot()}
    atomic_json(ACTIVE, payload)
    return payload


def finish(
    status: str,
    exit_code: int,
    next_run: str | None = None,
    message: str | None = None,
    duration_seconds: float | None = None,
) -> dict:
    active = read_json(ACTIVE)
    if not active:
        active = {"run_started_at": None, "before": snapshot()}
    after = snapshot()
    before = active["before"]
    started = datetime.fromisoformat(active["run_started_at"]) if active.get("run_started_at") else None
    ended = datetime.fromisoformat(after["captured_at"])
    observed_interval = (ended - started).total_seconds() if started else None
    duration = duration_seconds if duration_seconds is not None else observed_interval
    deltas = {
        "stage21_cached_images": after["stage21_cached_images"] - before["stage21_cached_images"],
        "stage22_steps": after["stage22_step"] - before["stage22_step"],
        "stage23_calibration_conditions": after["stage23_calibration_conditions"] - before["stage23_calibration_conditions"],
        "stage23_evaluation_conditions": after["stage23_evaluation_conditions"] - before["stage23_evaluation_conditions"],
    }
    report = {
        "run_started_at": active.get("run_started_at"),
        "run_finished_at": after["captured_at"],
        "duration_seconds": duration,
        "observed_start_to_report_seconds": observed_interval,
        "duration_source": "explicit worker evidence" if duration_seconds is not None else "start-to-report interval",
        "status": status,
        "exit_code": int(exit_code),
        "message": message,
        "next_run": next_run,
        "before": before,
        "after": after,
        "deltas": deltas,
        "isolation_passed": not after["test_pixels_extracted"] and not after["test_accessed"],
    }
    atomic_json(LATEST_JSON, report)
    with HISTORY.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
    duration_text = "unknown" if duration is None else f"{duration / 60:.2f} min"
    lines = [
        "# TokCom/TaiChi scheduled run report", "",
        f"- Status: **{status}** (exit code {exit_code})",
        f"- Started: {report['run_started_at']}",
        f"- Finished: {report['run_finished_at']}",
        f"- Duration: {duration_text}",
        f"- Next run: {next_run or 'not supplied'}", "",
        "## Work completed", "",
        f"- Stage 21 token cache: {before['stage21_cached_images']} -> {after['stage21_cached_images']} (delta {deltas['stage21_cached_images']:+d})",
        f"- Stage 22 training step: {before['stage22_step']} -> {after['stage22_step']} (delta {deltas['stage22_steps']:+d})",
        f"- Stage 23 calibration conditions: {before['stage23_calibration_conditions']} -> {after['stage23_calibration_conditions']} (delta {deltas['stage23_calibration_conditions']:+d})",
        f"- Stage 23 evaluation conditions: {before['stage23_evaluation_conditions']} -> {after['stage23_evaluation_conditions']} (delta {deltas['stage23_evaluation_conditions']:+d})", "",
        "## Isolation", "",
        f"- Official test pixels extracted: {str(after['test_pixels_extracted']).lower()}",
        f"- Official test accessed: {str(after['test_accessed']).lower()}",
        f"- Isolation audit: {'passed' if report['isolation_passed'] else 'FAILED'}",
    ]
    if message:
        lines.extend(["", "## Message", "", message])
    temporary = LATEST_MD.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, LATEST_MD)
    if ACTIVE.exists():
        ACTIVE.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("start")
    end = subparsers.add_parser("finish")
    end.add_argument("--status", choices=("success", "failed", "interrupted"), required=True)
    end.add_argument("--exit-code", type=int, required=True)
    end.add_argument("--next-run")
    end.add_argument("--message-env")
    end.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    if args.action == "start":
        print(json.dumps(start()))
    else:
        message = os.environ.get(args.message_env) if args.message_env else None
        print(json.dumps(finish(args.status, args.exit_code, args.next_run, message, args.duration_seconds)))


if __name__ == "__main__":
    main()
