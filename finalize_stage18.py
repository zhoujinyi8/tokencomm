"""Fail-closed Stage-18 audit, V3 comparison, and compact figures."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from context_maskgit_v3 import load_context_maskgit_v3
from train_stage15_contextual_tokcom import evaluate


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage18_strong_cmi"
FIGURES = ROOT / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate_conditions(entries: list[dict]) -> dict:
    output = {}
    for mode in ("correct", "wrong", "none"):
        rows = [condition for entry in entries for condition in entry["conditions"] if condition["mode"] == mode]
        output[mode] = {
            "ce_mean": float(np.mean([row["ce"] for row in rows])),
            "ce_std": float(np.std([row["ce"] for row in rows])),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
            "seeds": len(rows),
        }
    return output


@torch.inference_mode()
def evaluate_v3_baseline(protocol: dict, validation: dict) -> dict:
    model = load_context_maskgit_v3(torch.load(ROOT / "runs/stage15_contextual_tokcom/best.pt", weights_only=True)).eval()
    entries = [
        {
            "seed": seed,
            "conditions": evaluate(
                model,
                validation,
                float(protocol["validation"]["per"]),
                int(seed),
                int(protocol["validation"]["recovery_steps"]),
                1.0,
            ),
        }
        for seed in protocol["validation"]["seeds"]
    ]
    return {"checkpoint_sha256": sha256(ROOT / "runs/stage15_contextual_tokcom/best.pt"), "conditions": aggregate_conditions(entries)}


def audit() -> tuple[dict, list[dict], dict, dict]:
    protocol = json.loads((ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json").read_text(encoding="utf-8"))
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status.get("test_accessed") or status.get("stage14_holdout_used"):
        raise ValueError("Stage-18 status is incomplete or violates isolation")
    history = [json.loads(line) for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    planned = int(protocol["training"]["steps"])
    steps = [int(row["step"]) for row in history]
    if steps != list(range(1, planned + 1)):
        raise ValueError("Stage-18 history is not exactly continuous and unique")
    for row in history:
        for field in ("loss", "masked_ce", "context_prior_ce", "lr", "elapsed_seconds"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} at step {row['step']}")
    validation_every = int(protocol["training"]["validation_every"])
    validation_rows = [row for row in history if "validation" in row]
    expected_validation = list(range(validation_every, planned + 1, validation_every))
    if [row["step"] for row in validation_rows] != expected_validation:
        raise ValueError("Frozen validation schedule is incomplete")
    results = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
    if not results.get("completed") or results.get("test_accessed") or results.get("stage14_holdout_used"):
        raise ValueError("Stage-18 result isolation failed")
    best = min(
        validation_rows,
        key=lambda row: (
            row["validation"]["correct_context_mean_ce"],
            -row["validation"]["correct_context_mean_accuracy"],
        ),
    )
    if results["best_step"] != best["step"]:
        raise ValueError("Saved best step violates the frozen selection rule")
    best_checkpoint = torch.load(RUN / "best.pt", weights_only=True)
    if best_checkpoint["step"] != best["step"] or best_checkpoint["protocol_sha256"] != sha256(ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json"):
        raise ValueError("Best checkpoint identity mismatch")
    sweep_groups = {}
    for guidance in protocol["validation"]["guidance_sweep"]:
        entries = [row for row in results["guidance_sweep"] if row["guidance_scale"] == guidance]
        if len(entries) != len(protocol["validation"]["seeds"]):
            raise ValueError("Guidance sweep seed coverage mismatch")
        sweep_groups[str(guidance)] = aggregate_conditions(entries)
    audit_result = {
        "status": "passed",
        "history_steps": len(history),
        "history_continuous_unique": True,
        "validation_steps": expected_validation,
        "best_step": best["step"],
        "best_checkpoint_sha256": sha256(RUN / "best.pt"),
        "protocol_sha256": sha256(ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json"),
        "optimizer_scheduler_rng_recovery_present": all(key in torch.load(RUN / "recovery.pt", weights_only=True) for key in ("optimizer", "scheduler", "rng")),
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    return protocol, history, results, {"audit": audit_result, "guidance": sweep_groups}


def plot(history: list[dict], guidance: dict, v3: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    steps = np.array([row["step"] for row in history])
    masked = np.array([row["masked_ce"] for row in history])
    prior = np.array([row["context_prior_ce"] for row in history])
    window = 50
    kernel = np.ones(window) / window
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps[window - 1 :], np.convolve(masked, kernel, mode="valid"), label="masked CE (50-step mean)")
    axes[0].plot(steps[window - 1 :], np.convolve(prior, kernel, mode="valid"), label="class-prior CE (50-step mean)")
    axes[0].set(xlabel="Step", ylabel="Cross entropy", title="Stage 18 training")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    validation = [row for row in history if "validation" in row]
    val_steps = [row["step"] for row in validation]
    for index, mode in enumerate(("correct", "wrong", "none")):
        ce = [float(np.mean([seed["conditions"][index]["ce"] for seed in row["validation"]["seeds"]])) for row in validation]
        axes[1].plot(val_steps, ce, marker="o", label=mode)
    axes[1].set(xlabel="Step", ylabel="Masked-token CE", title="Frozen validation controls")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    figure.savefig(FIGURES / "stage18_training_and_validation.png", dpi=180)
    plt.close(figure)

    scales = [float(key) for key in guidance]
    correct_ce = [guidance[str(scale)]["correct"]["ce_mean"] for scale in scales]
    wrong_ce = [guidance[str(scale)]["wrong"]["ce_mean"] for scale in scales]
    none_ce = [guidance[str(scale)]["none"]["ce_mean"] for scale in scales]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(scales, correct_ce, marker="o", label="V4 correct")
    axis.plot(scales, wrong_ce, marker="o", label="V4 wrong")
    axis.plot(scales, none_ce, marker="o", label="V4 none")
    axis.axhline(v3["conditions"]["correct"]["ce_mean"], linestyle="--", color="black", label="V3 correct, scale 1")
    axis.set(xlabel="Guidance scale", ylabel="Masked-token CE", title="Stage 18 guidance and V3 baseline")
    axis.grid(alpha=0.25); axis.legend()
    figure.savefig(FIGURES / "stage18_guidance_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.set_num_threads(4)
    protocol, history, results, derived = audit()
    validation = torch.load(ROOT / "data/imagenette_stage10/token_cache/val.pt", weights_only=True)
    v3 = evaluate_v3_baseline(protocol, validation)
    plot(history, derived["guidance"], v3)
    output = {
        "stage": 18,
        "status": "completed",
        "evidence_level": protocol["evidence_level"],
        "best_step": results["best_step"],
        "best_correct_context_mean_ce": results["best_correct_context_mean_ce"],
        "best_correct_context_mean_accuracy": results["best_correct_context_mean_accuracy"],
        "guidance": derived["guidance"],
        "v3_baseline_on_stage18_seeds": v3,
        "audit": derived["audit"],
        "figures": ["figures/stage18_training_and_validation.png", "figures/stage18_guidance_comparison.png"],
    }
    atomic_json(ROOT / "STAGE18_SUMMARY.json", output)
    atomic_json(ROOT / "STAGE18_AUDIT.json", derived["audit"])
    print(json.dumps(derived["audit"]))


if __name__ == "__main__":
    main()
