"""Fail-closed Stage-22 audit, V4 baseline, summaries, and figures."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from context_maskgit_v4 import load_context_maskgit_v4
from models import erase_packets
from train_stage22_contextual_tokcom import ROOT, balanced_indices, read_jsonl, sha256_file


RUN = ROOT / "runs/stage22_contextual_tokcom"
CACHE = ROOT / "data/stage20_food101/token_cache_stage21"
FIGURES = ROOT / "figures"
PROTOCOL_PATH = ROOT / "STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate_seed_conditions(entries: list[dict]) -> dict:
    output = {}
    for mode in ("correct", "wrong", "none"):
        rows = [condition for entry in entries for condition in entry["conditions"] if condition["mode"] == mode]
        if len(rows) != len(entries):
            raise ValueError(f"Stage-22 condition coverage mismatch for {mode}")
        output[mode] = {
            "masked_ce_mean": float(np.mean([row["masked_ce"] for row in rows])),
            "masked_ce_std": float(np.std([row["masked_ce"] for row in rows])),
            "one_pass_accuracy_mean": float(np.mean([row["one_pass_accuracy"] for row in rows])),
            "one_pass_accuracy_std": float(np.std([row["one_pass_accuracy"] for row in rows])),
            "seeds": len(rows),
        }
    return output


def audit() -> tuple[dict, list[dict], dict, dict]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("test_accessed") or protocol.get("test_pixels_extracted"):
        raise ValueError("Stage-22 protocol is incomplete or violates isolation")
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Stage-22 frozen input changed: {name}")
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    results = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status.get("test_accessed"):
        raise ValueError("Stage-22 status is incomplete or violates isolation")
    if not results.get("completed") or results.get("test_accessed"):
        raise ValueError("Stage-22 results are incomplete or violate isolation")
    history = [json.loads(line) for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    planned_steps = int(protocol["training"]["steps"])
    steps = [int(row["step"]) for row in history]
    if steps != list(range(1, planned_steps + 1)):
        raise ValueError("Stage-22 history is not exactly continuous and unique")
    for row in history:
        for field in ("loss", "masked_ce", "context_prior_ce", "lr", "elapsed_seconds"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite Stage-22 {field} at step {row['step']}")
    validation_every = int(protocol["training"]["validation_every"])
    expected_validation_steps = list(range(validation_every, planned_steps + 1, validation_every))
    validation_rows = [row for row in history if "validation" in row]
    if [int(row["step"]) for row in validation_rows] != expected_validation_steps:
        raise ValueError("Stage-22 frozen validation schedule is incomplete")
    best = min(
        validation_rows,
        key=lambda row: (
            float(row["validation"]["correct_context_mean_ce"]),
            -float(row["validation"]["correct_context_mean_accuracy"]),
        ),
    )
    if int(results["best_step"]) != int(best["step"]) or int(status["best_step"]) != int(best["step"]):
        raise ValueError("Stage-22 best step violates the frozen selection rule")
    protocol_hash = sha256_file(PROTOCOL_PATH)
    best_checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    recovery = torch.load(RUN / "recovery.pt", map_location="cpu", weights_only=True)
    if best_checkpoint.get("model") != "context_maskgit_v5" or int(best_checkpoint["step"]) != int(best["step"]):
        raise ValueError("Stage-22 best checkpoint identity mismatch")
    if best_checkpoint.get("protocol_sha256") != protocol_hash or recovery.get("protocol_sha256") != protocol_hash:
        raise ValueError("Stage-22 checkpoint protocol identity mismatch")
    if not all(key in recovery for key in ("optimizer", "scheduler", "rng")):
        raise ValueError("Stage-22 recovery state is incomplete")
    final_entries = results.get("final_validation", [])
    if [int(entry["seed"]) for entry in final_entries] != [int(seed) for seed in protocol["validation"]["seeds"]]:
        raise ValueError("Stage-22 final validation seed coverage mismatch")
    final_controls = aggregate_seed_conditions(final_entries)
    audit_payload = {
        "status": "passed",
        "history_steps": len(history),
        "history_continuous_unique": True,
        "validation_steps": expected_validation_steps,
        "chunk_boundary_evaluations": sum("chunk_boundary_validation" in row for row in history),
        "best_step": int(best["step"]),
        "best_checkpoint_sha256": sha256_file(RUN / "best.pt"),
        "protocol_sha256": protocol_hash,
        "pre_registered_plan_sha256": protocol["pre_registered_plan_sha256"],
        "optimizer_scheduler_rng_recovery_present": True,
        "official_test_images_sealed": int(protocol["data"]["official_test_images_sealed"]),
        "test_pixels_extracted": False,
        "test_accessed": False,
    }
    return protocol, history, results, {"audit": audit_payload, "final_controls": final_controls}


@torch.inference_mode()
def evaluate_v4_unconditional(protocol: dict) -> dict:
    dev = torch.load(CACHE / "dev.pt", map_location="cpu", weights_only=True)
    records = read_jsonl(ROOT / "data/stage20_food101/dev.jsonl")
    indices = balanced_indices(
        records,
        int(protocol["validation"]["images_per_class"]),
        int(protocol["validation"]["subset_seed"]),
    )
    ids = dev["ids"][indices].long()
    model = load_context_maskgit_v4(torch.load(ROOT / protocol["initialization"]["checkpoint"], map_location="cpu", weights_only=True)).eval()
    rows = []
    for seed in protocol["validation"]["seeds"]:
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            missing = erase_packets(ids, float(protocol["validation"]["per"]), int(protocol["training"]["packet_tokens"]))
        total_ce = total_correct = total_missing = 0
        batch_size = int(protocol["validation"]["batch_size"])
        for start in range(0, len(ids), batch_size):
            target = ids[start : start + batch_size]
            mask = missing[start : start + batch_size]
            logits = model(target.masked_fill(mask, model.codes), None)
            total_ce += float(F.cross_entropy(logits[mask], target[mask], reduction="sum"))
            total_correct += int((logits[mask].argmax(-1) == target[mask]).sum())
            total_missing += int(mask.sum())
        rows.append({
            "seed": int(seed),
            "masked_ce": total_ce / total_missing,
            "one_pass_accuracy": total_correct / total_missing,
            "missing_tokens": total_missing,
        })
    return {
        "checkpoint": protocol["initialization"]["checkpoint"],
        "checkpoint_sha256": protocol["initialization"]["checkpoint_sha256"],
        "masked_ce_mean": float(np.mean([row["masked_ce"] for row in rows])),
        "masked_ce_std": float(np.std([row["masked_ce"] for row in rows])),
        "one_pass_accuracy_mean": float(np.mean([row["one_pass_accuracy"] for row in rows])),
        "one_pass_accuracy_std": float(np.std([row["one_pass_accuracy"] for row in rows])),
        "seeds": rows,
    }


def plot(history: list[dict], final_controls: dict, v4: dict) -> list[str]:
    FIGURES.mkdir(exist_ok=True)
    steps = np.array([row["step"] for row in history])
    loss = np.array([row["loss"] for row in history])
    window = 100
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps[window - 1 :], np.convolve(loss, np.ones(window) / window, mode="valid"))
    axes[0].set(xlabel="Step", ylabel="Total loss", title="Stage 22 training (100-step mean)")
    axes[0].grid(alpha=0.25)
    validation = [row for row in history if "validation" in row]
    for mode_index, mode in enumerate(("correct", "wrong", "none")):
        values = [
            float(np.mean([entry["conditions"][mode_index]["masked_ce"] for entry in row["validation"]["seeds"]]))
            for row in validation
        ]
        axes[1].plot([row["step"] for row in validation], values, marker="o", label=mode)
    axes[1].set(xlabel="Step", ylabel="Masked-token CE", title="Frozen 1,010-image validation")
    axes[1].grid(alpha=0.25); axes[1].legend()
    training_path = FIGURES / "stage22_training_and_validation.png"
    figure.savefig(training_path, dpi=180)
    plt.close(figure)

    names = ["V5 correct", "V5 wrong", "V5 none", "V4 none"]
    ce = [final_controls[mode]["masked_ce_mean"] for mode in ("correct", "wrong", "none")] + [v4["masked_ce_mean"]]
    accuracy = [final_controls[mode]["one_pass_accuracy_mean"] for mode in ("correct", "wrong", "none")] + [v4["one_pass_accuracy_mean"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(names, ce); axes[0].set(ylabel="Masked-token CE", title="Context controls")
    axes[1].bar(names, np.array(accuracy) * 100); axes[1].set(ylabel="One-pass accuracy (%)", title="Context controls")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20); axis.grid(axis="y", alpha=0.25)
    controls_path = FIGURES / "stage22_context_controls.png"
    figure.savefig(controls_path, dpi=180)
    plt.close(figure)
    return [str(training_path.relative_to(ROOT)).replace("\\", "/"), str(controls_path.relative_to(ROOT)).replace("\\", "/")]


def main() -> None:
    torch.set_num_threads(4)
    protocol, history, results, derived = audit()
    v4 = evaluate_v4_unconditional(protocol)
    figures = plot(history, derived["final_controls"], v4)
    summary = {
        "stage": 22,
        "status": "completed",
        "evidence_level": protocol["evidence_level"],
        "best_step": results["best_step"],
        "best_correct_context_mean_ce": results["best_correct_context_mean_ce"],
        "best_correct_context_mean_accuracy": results["best_correct_context_mean_accuracy"],
        "final_controls": derived["final_controls"],
        "v4_unconditional_baseline": v4,
        "audit": derived["audit"],
        "figures": figures,
        "limitations": protocol["paper_alignment"]["approximate"],
    }
    atomic_json(ROOT / "STAGE22_SUMMARY.json", summary)
    atomic_json(ROOT / "STAGE22_AUDIT.json", derived["audit"])
    print(json.dumps(derived["audit"]))


if __name__ == "__main__":
    main()
