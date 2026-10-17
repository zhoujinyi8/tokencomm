"""Fail-closed Stage-23 audit, aggregate report, and link trade-off figures."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_stage23_v5_link import ROOT, condition_key, read_jsonl, sha256_file, summarize
from stage23_link_common import balanced_window_indices, select_threshold


RUN = ROOT / "runs/stage23_v5_link"
PROTOCOL_PATH = ROOT / "STAGE23_V5_LINK_PROTOCOL.json"
FIGURES = ROOT / "figures"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit() -> tuple[dict, list[dict], list[dict], dict, list[dict]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("test_accessed") or protocol.get("test_pixels_extracted"):
        raise ValueError("Stage-23 protocol is incomplete or contaminated")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-23 implementation changed: {name}")
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    calibration = json.loads((RUN / "calibration_results.json").read_text(encoding="utf-8"))
    evaluation = json.loads((RUN / "evaluation_results.json").read_text(encoding="utf-8"))
    selection_payload = json.loads((RUN / "threshold_selection.json").read_text(encoding="utf-8"))
    runtime_summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status.get("test_accessed") or selection_payload.get("test_accessed") or runtime_summary.get("test_accessed"):
        raise ValueError("Stage-23 runtime status is incomplete or violates isolation")
    expected_calibration = {condition_key(row) for row in protocol["calibration_conditions"]}
    expected_evaluation = {condition_key(row) for row in protocol["evaluation_conditions"]}
    actual_calibration = [condition_key(row) for row in calibration]
    actual_evaluation = [condition_key(row) for row in evaluation]
    if len(actual_calibration) != len(set(actual_calibration)) or set(actual_calibration) != expected_calibration:
        raise ValueError("Stage-23 calibration condition coverage mismatch")
    if len(actual_evaluation) != len(set(actual_evaluation)) or set(actual_evaluation) != expected_evaluation:
        raise ValueError("Stage-23 evaluation condition coverage mismatch")
    for row in calibration + evaluation:
        if row.get("test_accessed") or row.get("images") != 505 or not row.get("restored_matches_shape"):
            raise ValueError("Stage-23 row identity or isolation mismatch")
        for value in row["metrics"].values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError("Stage-23 contains a non-finite metric")
    recomputed_selection = {}
    for channel in protocol["link"]["channels"]:
        name = channel["channel"]
        rows = [row for row in calibration if row["channel"] == name]
        recomputed_selection[name] = select_threshold(rows, protocol["link"]["threshold_candidates"], float(protocol["threshold_selection"]["target_recovery_fraction"]))
    if selection_payload.get("selection") != recomputed_selection or selection_payload.get("evaluation_subset_used_for_selection") is not False:
        raise ValueError("Stage-23 threshold selection violates the pre-registered rule")
    recomputed_summary = summarize(evaluation)
    if runtime_summary.get("summary") != recomputed_summary or runtime_summary.get("selection") != recomputed_selection:
        raise ValueError("Stage-23 aggregate summary did not recompute exactly")
    records = read_jsonl(ROOT / protocol["development_records"])
    seed = int(protocol["data"]["ranking_seed"])
    model_selection = set(balanced_window_indices(records, start_per_class=0, images_per_class=10, seed=seed))
    threshold_calibration = set(balanced_window_indices(records, start_per_class=10, images_per_class=5, seed=seed))
    link_evaluation = set(balanced_window_indices(records, start_per_class=15, images_per_class=5, seed=seed))
    if not model_selection.isdisjoint(threshold_calibration) or not model_selection.isdisjoint(link_evaluation) or not threshold_calibration.isdisjoint(link_evaluation):
        raise ValueError("Stage-23 development subsets overlap")
    audit_payload = {
        "status": "passed", "calibration_conditions": len(calibration), "evaluation_conditions": len(evaluation),
        "calibration_images": len(threshold_calibration), "evaluation_images": len(link_evaluation),
        "subsets_disjoint": True, "selection_recomputed": True, "aggregates_recomputed": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "calibration_results_sha256": sha256_file(RUN / "calibration_results.json"),
        "evaluation_results_sha256": sha256_file(RUN / "evaluation_results.json"),
        "official_test_images_sealed": int(protocol["official_test_images_sealed"]), "test_pixels_extracted": False, "test_accessed": False,
    }
    return protocol, calibration, evaluation, recomputed_selection, recomputed_summary


def plot(protocol: dict, calibration: list[dict], summary: list[dict], selection: dict) -> list[str]:
    FIGURES.mkdir(exist_ok=True)
    paths = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, channel in zip(axes, ("awgn", "rayleigh")):
        points = []
        for threshold in protocol["link"]["threshold_candidates"]:
            mode = f"selective_{threshold:g}"
            rows = [row for row in calibration if row["channel"] == channel and row["mode"] == mode]
            points.append((threshold, np.mean([row["metrics"]["token_error_rate"] for row in rows]), np.mean([row["metrics"]["tce_pixels_per_token_bit"] for row in rows])))
        axis.plot([point[2] for point in points], [100 * point[1] for point in points], marker="o")
        chosen = selection[channel]
        axis.scatter([chosen["tce"]], [100 * chosen["token_error_rate"]], marker="*", s=180, label=f"selected {chosen['threshold']:g}")
        axis.set(xlabel="TCE (pixels/token-bit)", ylabel="Token error rate (%)", title=f"{channel.upper()} calibration")
        axis.grid(alpha=0.25); axis.legend()
    calibration_path = FIGURES / "stage23_threshold_calibration.png"
    figure.savefig(calibration_path, dpi=180); plt.close(figure)
    paths.append(str(calibration_path.relative_to(ROOT)).replace("\\", "/"))

    modes = ["no_cmi", "cmi_no_arq", "cmi_selective", "full_arq"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, channel in zip(axes, ("awgn", "rayleigh")):
        rows = {row["mode"]: row for row in summary if row["channel"] == channel}
        axis.scatter([rows[mode]["tce_pixels_per_token_bit_mean"] for mode in modes], [100 * rows[mode]["recovered_token_accuracy_mean"] for mode in modes])
        for mode in modes:
            axis.annotate(mode, (rows[mode]["tce_pixels_per_token_bit_mean"], 100 * rows[mode]["recovered_token_accuracy_mean"]), fontsize=8)
        axis.set(xlabel="TCE (pixels/token-bit)", ylabel="Recovered token accuracy (%)", title=f"{channel.upper()} held development")
        axis.grid(alpha=0.25)
    evaluation_path = FIGURES / "stage23_link_tradeoff.png"
    figure.savefig(evaluation_path, dpi=180); plt.close(figure)
    paths.append(str(evaluation_path.relative_to(ROOT)).replace("\\", "/"))
    return paths


def main() -> None:
    protocol, calibration, evaluation, selection, summary = audit()
    figures = plot(protocol, calibration, summary, selection)
    audit_payload = {
        "status": "passed", "calibration_conditions": len(calibration), "evaluation_conditions": len(evaluation),
        "subsets_disjoint": True, "selection_recomputed": True, "aggregates_recomputed": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "calibration_results_sha256": sha256_file(RUN / "calibration_results.json"),
        "evaluation_results_sha256": sha256_file(RUN / "evaluation_results.json"),
        "official_test_images_sealed": int(protocol["official_test_images_sealed"]), "test_pixels_extracted": False, "test_accessed": False,
    }
    atomic_json(ROOT / "STAGE23_AUDIT.json", audit_payload)
    atomic_json(ROOT / "STAGE23_V5_LINK_SUMMARY.json", {
        "stage": 23, "status": "completed", "evidence_level": protocol["evidence_level"],
        "selection": selection, "summary": summary, "audit": audit_payload, "figures": figures,
    })
    print(json.dumps(audit_payload))


if __name__ == "__main__":
    main()
