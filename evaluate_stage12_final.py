"""Run the exact resumable conditions frozen in STAGE12_FINAL_TEST_PROTOCOL.json."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from prepare_stage12_holdout import ROOT, sha256, validate_frozen_protocol
from qwen_dual_taichi import QwenDualTaiChi
from train_qwen_dual_taichi import batch


def condition_key(row):
    return row["variant"], row["kind"], float(row["snr_db"]), int(row["seed"])


def planned_conditions(protocol):
    planned = [
        {"variant": "correct", "kind": kind, "snr_db": float(snr), "seed": int(seed)}
        for kind in protocol["channels"]
        for snr in protocol["snr_db"]
        for seed in protocol["seeds"]
    ]
    planned.extend(
        {
            "variant": control["variant"],
            "kind": control["channel"],
            "snr_db": float(control["snr_db"]),
            "seed": int(control["seed"]),
        }
        for control in protocol["controls"]
    )
    keys = [condition_key(row) for row in planned]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen protocol contains duplicate evaluation conditions")
    return planned


@torch.inference_mode()
def evaluate(model, data, indices, snr, kind, seed, variant="correct"):
    model.eval()
    predictions, targets, labels = [], [], []
    total_loss = 0.0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0, len(indices), 2):
            chosen = indices[start : start + 2]
            images, questions, answers = batch(data, chosen, variant)
            output = model(images, questions, answers, snr, kind)
            total_loss += float(output.loss) * len(answers)
            predictions.extend(model.generate_answers(images, questions, snr, kind, max_new_tokens=12))
            targets.extend(answers)
            labels.extend(int(data["labels"][index]) for index in chosen)
    correct = [prediction.lower() == target.lower() for prediction, target in zip(predictions, targets)]
    classes = sorted(set(labels))
    per_class = {
        str(label): sum(ok for ok, value in zip(correct, labels) if value == label) / sum(value == label for value in labels)
        for label in classes
    }
    return {
        "kind": kind,
        "snr_db": float(snr),
        "seed": int(seed),
        "variant": variant,
        "count": len(indices),
        "loss": total_loss / len(indices),
        "exact_match": sum(correct) / len(correct),
        "per_class_accuracy": per_class,
        "error_indices": [indices[i] for i, ok in enumerate(correct) if not ok],
    }


def summarize(rows, protocol):
    planned = planned_conditions(protocol)
    planned_keys = {condition_key(row) for row in planned}
    actual_keys = [condition_key(row) for row in rows]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("Duplicate result condition")
    if not set(actual_keys).issubset(planned_keys):
        raise ValueError("Result contains a condition not present in the frozen protocol")
    summary = []
    for kind in protocol["channels"]:
        for snr in protocol["snr_db"]:
            group = [
                row for row in rows
                if row["variant"] == "correct" and row["kind"] == kind
                and float(row["snr_db"]) == float(snr) and int(row["seed"]) in protocol["seeds"]
            ]
            if len(group) != len(protocol["seeds"]):
                continue
            values = torch.tensor([row["exact_match"] for row in group])
            losses = torch.tensor([row["loss"] for row in group])
            summary.append(
                {
                    "kind": kind,
                    "snr_db": float(snr),
                    "seeds": len(group),
                    "exact_match_mean": float(values.mean()),
                    "exact_match_std": float(values.std(unbiased=False)),
                    "loss_mean": float(losses.mean()),
                }
            )
    controls_keys = {
        (row["variant"], row["channel"], float(row["snr_db"]), int(row["seed"]))
        for row in protocol["controls"]
    }
    controls = [row for row in rows if condition_key(row) in controls_keys]
    return summary, controls, len(actual_keys) == len(planned_keys)


def main() -> None:
    protocol_path = ROOT / "STAGE12_FINAL_TEST_PROTOCOL.json"
    if not protocol_path.exists():
        raise FileNotFoundError("Frozen Stage-12 protocol does not exist; test remains sealed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol, protocol_path)
    checkpoint = ROOT / protocol["checkpoint"]
    if sha256(checkpoint).lower() != protocol["checkpoint_sha256"].lower():
        raise ValueError("Checkpoint no longer matches frozen protocol")
    llm_path = Path(protocol["llm_path"])
    if sha256(llm_path / "model.safetensors").lower() != protocol["llm_weight_sha256"].lower():
        raise ValueError("Frozen Qwen base weight hash mismatch")
    cache_path = ROOT / protocol["feature_cache"]
    data = torch.load(cache_path, weights_only=True)
    identity = data["identity"]
    if identity["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol/cache mismatch")
    if identity["checkpoint_sha256"].lower() != protocol["checkpoint_sha256"].lower():
        raise ValueError("Checkpoint/cache mismatch")
    if identity["model_evaluation_accessed"] is not False or identity["used_for_training"] is not False:
        raise ValueError("Test contamination marker")
    if len(data["labels"]) != protocol["expected_test_images"]:
        raise ValueError("Test count mismatch")

    torch.set_num_threads(4)
    model = QwenDualTaiChi(
        llm_path,
        lora_rank=int(protocol["lora_rank"]),
        fusion_variant=protocol["selected_variant"],
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    if set(checkpoint_data["trainable_state"]) != {name for name, parameter in model.named_parameters() if parameter.requires_grad}:
        raise ValueError("Checkpoint trainable parameter set does not match frozen model")
    for name, value in checkpoint_data["trainable_state"].items():
        named[name].data.copy_(value)

    output = ROOT / "runs" / "stage12_final_test"
    output.mkdir(parents=True, exist_ok=True)
    started = {
        "protocol_sha256": sha256(protocol_path),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "test_accessed": True,
        "used_for_training": False,
    }
    (output / "access.json").write_text(json.dumps(started, indent=2), encoding="utf-8")
    (ROOT / protocol["holdout_directory"] / "evaluation_access.json").write_text(
        json.dumps(started, indent=2), encoding="utf-8"
    )
    results_path = output / "results.jsonl"
    rows = []
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    plan = planned_conditions(protocol)
    plan_keys = {condition_key(row) for row in plan}
    completed = {condition_key(row): row for row in rows}
    if len(completed) != len(rows) or not set(completed).issubset(plan_keys):
        raise ValueError("Existing Stage-12 results violate frozen plan")
    indices = list(range(len(data["labels"])))
    with results_path.open("a", encoding="utf-8", buffering=1) as stream:
        for condition in plan:
            key = condition_key(condition)
            if key in completed:
                continue
            row = evaluate(
                model,
                data,
                indices,
                condition["snr_db"],
                condition["kind"],
                condition["seed"],
                condition["variant"],
            )
            stream.write(json.dumps(row) + "\n")
            print(row, flush=True)
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    summary, controls, complete = summarize(rows, protocol)
    if not complete:
        raise RuntimeError("Not all frozen Stage-12 conditions completed")
    final = {
        "protocol_sha256": sha256(protocol_path),
        "checkpoint_step": checkpoint_data["step"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "selected_variant": protocol["selected_variant"],
        "test_images": len(indices),
        "test_accessed": True,
        "used_for_training": False,
        "conditions_completed": len(rows),
        "summary": summary,
        "controls": controls,
    }
    (output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (output / "status.json").write_text(
        json.dumps({"completed": True, "test_images": len(indices), "checkpoint_step": checkpoint_data["step"]}),
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
