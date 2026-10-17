"""Resumable token-link calibration and held-development evaluation for V5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from context_maskgit_v5 import load_context_maskgit_v5
from stage23_link_common import balanced_window_indices, select_threshold
from tokcom_arq import transmit_tokens_with_arq


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE23_V5_LINK_PROTOCOL.json"
OUT = ROOT / "runs/stage23_v5_link"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def condition_key(row: dict) -> tuple[str, float, int, str]:
    return row["channel"], float(row["snr_esn0_db"]), int(row["seed"]), row["mode"]


def validate(protocol: dict) -> tuple[torch.nn.Module, dict, list[dict]]:
    if protocol.get("status") != "frozen" or protocol.get("test_accessed") or protocol.get("test_pixels_extracted"):
        raise ValueError("Stage-23 protocol is not safely frozen")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-23 implementation changed: {name}")
    if sha256_file(ROOT / "STAGE23_V5_LINK_PLAN.json") != protocol["pre_registered_plan_sha256"]:
        raise ValueError("Stage-23 pre-registered plan changed")
    if sha256_file(ROOT / "STAGE22_SUMMARY.json") != protocol["stage22_summary_sha256"]:
        raise ValueError("Stage-22 summary changed after Stage-23 freeze")
    for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("development_cache", "development_cache_sha256"), ("development_records", "development_records_sha256")):
        if sha256_file(ROOT / protocol[field]) != protocol[hash_field]:
            raise ValueError(f"Frozen Stage-23 input changed: {field}")
    audit = json.loads((ROOT / "STAGE22_AUDIT.json").read_text(encoding="utf-8"))
    if sha256_file(ROOT / "STAGE22_AUDIT.json") != protocol["stage22_audit_sha256"] or audit.get("status") != "passed" or audit.get("test_accessed") or audit.get("test_pixels_extracted"):
        raise ValueError("Stage-22 audit identity or isolation failed")
    checkpoint = torch.load(ROOT / protocol["checkpoint"], map_location="cpu", weights_only=True)
    model = load_context_maskgit_v5(checkpoint).eval()
    data = torch.load(ROOT / protocol["development_cache"], map_location="cpu", weights_only=True)
    records = read_jsonl(ROOT / protocol["development_records"])
    labels = torch.tensor([row["label"] for row in records], dtype=torch.int16)
    if data["ids"].shape != (5050, 256) or not torch.equal(data["labels"], labels):
        raise ValueError("Stage-23 development cache is misaligned")
    return model, data, records


def subset(protocol: dict, data: dict, records: list[dict], phase: str) -> tuple[torch.Tensor, torch.Tensor]:
    field = "threshold_calibration_window_per_class" if phase == "calibration" else "link_evaluation_window_per_class"
    start, stop = protocol["data"][field]
    indices = balanced_window_indices(records, start_per_class=int(start), images_per_class=int(stop - start), seed=int(protocol["data"]["ranking_seed"]))
    index = torch.tensor(indices, dtype=torch.long)
    return data["ids"][index].long(), data["labels"][index].long()


def parameters(protocol: dict, condition: dict, selection: dict | None) -> tuple[str, float, bool, float]:
    mode = condition["mode"]
    if mode in {"semantic_only", "cmi_no_arq"}:
        return "confidence", 0.0, True, 1.0
    if mode == "no_cmi":
        return "confidence", 0.0, False, 0.0
    if mode == "full_arq":
        return "all_failed", 1.0, True, 1.0
    if mode == "cmi_selective":
        if selection is None:
            raise ValueError("Selective evaluation requires frozen calibration selection")
        chosen = selection[condition["channel"]]
        return "confidence", float(chosen["threshold"]), True, 1.0
    if mode.startswith("selective_"):
        return "confidence", float(mode.removeprefix("selective_")), True, 1.0
    raise ValueError(f"Unknown Stage-23 mode: {mode}")


def run_condition(model, ids: torch.Tensor, labels: torch.Tensor, protocol: dict, condition: dict, selection: dict | None) -> dict:
    policy, threshold, use_context, guidance = parameters(protocol, condition, selection)
    restored, missing, metrics = transmit_tokens_with_arq(
        ids.numpy(), receiver=model, labels=labels if use_context else None, codebook_size=1024,
        snr_db=float(condition["snr_esn0_db"]), channel=condition["channel"], seed=int(condition["seed"]),
        packet_tokens=int(protocol["link"]["packet_tokens"]), max_rounds=int(protocol["link"]["max_rounds"]),
        policy=policy, confidence_threshold=threshold, recovery_steps=int(protocol["link"]["recovery_steps"]),
        guidance_scale=guidance,
    )
    if not all(math.isfinite(float(value)) for key, value in metrics.items() if isinstance(value, (int, float))):
        raise RuntimeError("Non-finite Stage-23 communication metric")
    return {
        **condition, "images": len(ids), "tokens": int(ids.numel()), "confidence_threshold": threshold,
        "metrics": metrics, "restored_matches_shape": list(restored.shape) == list(ids.shape),
        "remaining_erasure_count": int(missing.sum()), "test_accessed": False,
    }


def derive_selection(protocol: dict, rows: list[dict]) -> dict:
    output = {}
    for channel in protocol["link"]["channels"]:
        name = channel["channel"]
        chosen = [row for row in rows if row["channel"] == name]
        output[name] = select_threshold(chosen, protocol["link"]["threshold_candidates"], float(protocol["threshold_selection"]["target_recovery_fraction"]))
    return output


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for channel in sorted({row["channel"] for row in rows}):
        for mode in sorted({row["mode"] for row in rows if row["channel"] == channel}):
            selected = [row for row in rows if row["channel"] == channel and row["mode"] == mode]
            metrics = [row["metrics"] for row in selected]
            output.append({
                "channel": channel, "snr_esn0_db": selected[0]["snr_esn0_db"], "mode": mode, "seeds": len(selected),
                **{f"{name}_mean": float(np.mean([item[name] for item in metrics])) for name in (
                    "token_error_rate", "recovered_token_accuracy", "tce_pixels_per_token_bit", "transmissions_per_packet", "final_erasure_rate")},
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Soft wall-clock budget; zero runs through all pending conditions")
    args = parser.parse_args()
    if args.threads <= 0 or args.max_seconds < 0:
        parser.error("threads must be positive and max-seconds nonnegative")
    torch.set_num_threads(args.threads)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    model, data, records = validate(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def budget_allows_another(last_duration: float) -> bool:
        if args.max_seconds == 0:
            return True
        reserve = max(60.0, 1.5 * last_duration)
        return time.monotonic() - started + reserve < args.max_seconds

    calibration_path = OUT / "calibration_results.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else []
    expected_calibration = {condition_key(row) for row in protocol["calibration_conditions"]}
    if len({condition_key(row) for row in calibration}) != len(calibration) or not {condition_key(row) for row in calibration}.issubset(expected_calibration):
        raise ValueError("Stage-23 calibration results contain duplicate or unplanned rows")
    pending = [row for row in protocol["calibration_conditions"] if condition_key(row) not in {condition_key(item) for item in calibration}]
    while pending:
        ids, labels = subset(protocol, data, records, "calibration")
        condition_started = time.monotonic()
        calibration.append(run_condition(model, ids, labels, protocol, pending[0], None))
        atomic_json(calibration_path, calibration)
        print(json.dumps({"phase": "calibration", "completed": len(calibration), "planned": len(expected_calibration)}))
        duration = time.monotonic() - condition_started
        pending = [row for row in protocol["calibration_conditions"] if condition_key(row) not in {condition_key(item) for item in calibration}]
        if pending and not budget_allows_another(duration):
            return
    selection_path = OUT / "threshold_selection.json"
    selection = derive_selection(protocol, calibration)
    atomic_json(selection_path, {"selection": selection, "calibration_rows": len(calibration), "evaluation_subset_used_for_selection": False, "test_accessed": False})
    evaluation_path = OUT / "evaluation_results.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else []
    expected_evaluation = {condition_key(row) for row in protocol["evaluation_conditions"]}
    if len({condition_key(row) for row in evaluation}) != len(evaluation) or not {condition_key(row) for row in evaluation}.issubset(expected_evaluation):
        raise ValueError("Stage-23 evaluation results contain duplicate or unplanned rows")
    pending = [row for row in protocol["evaluation_conditions"] if condition_key(row) not in {condition_key(item) for item in evaluation}]
    if pending and args.max_seconds > 0 and time.monotonic() - started + 60.0 >= args.max_seconds:
        return
    while pending:
        ids, labels = subset(protocol, data, records, "evaluation")
        condition_started = time.monotonic()
        evaluation.append(run_condition(model, ids, labels, protocol, pending[0], selection))
        atomic_json(evaluation_path, evaluation)
        print(json.dumps({"phase": "evaluation", "completed": len(evaluation), "planned": len(expected_evaluation)}))
        duration = time.monotonic() - condition_started
        pending = [row for row in protocol["evaluation_conditions"] if condition_key(row) not in {condition_key(item) for item in evaluation}]
        if pending and not budget_allows_another(duration):
            return
    atomic_json(OUT / "summary.json", {"conditions": len(evaluation), "selection": selection, "summary": summarize(evaluation), "test_accessed": False})
    atomic_json(OUT / "status.json", {"completed": True, "calibration_conditions": len(calibration), "evaluation_conditions": len(evaluation), "test_accessed": False})
    print(json.dumps({"completed": True, "calibration_conditions": len(calibration), "evaluation_conditions": len(evaluation)}))


if __name__ == "__main__":
    main()
