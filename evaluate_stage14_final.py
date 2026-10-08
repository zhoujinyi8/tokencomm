"""Resumable preregistered Stage-14 TokCom blind evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from context_maskgit_v2 import load_context_maskgit_v2
from digital_link import transmit_tokens
from evaluate_stage13_physical import key, sha256
from run_pretrained import spatial_fill


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE14_TOKCOM_FINAL_PROTOCOL.json"
OUT = ROOT / "runs/stage14_tokcom_final"


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate(protocol: dict):
    if protocol.get("status") != "frozen" or protocol.get("test_accessed_at_freeze") is not False:
        raise ValueError("Stage-14 protocol is not sealed")
    checkpoint = ROOT / protocol["checkpoint"]
    cache_path = ROOT / protocol["token_cache"]
    holdout = ROOT / protocol["holdout_directory"]
    if sha256(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((holdout / "identity.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL_PATH) or identity["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise ValueError("Holdout was not materialized from this protocol")
    if sha256(holdout / "test.jsonl") != manifest["test_manifest_sha256"]:
        raise ValueError("Blind-test manifest changed")
    cache = torch.load(cache_path, weights_only=True)
    if cache["identity"]["manifest_sha256"] != sha256(holdout / "manifest.json"):
        raise ValueError("Token cache is not bound to the blind holdout")
    if cache["identity"]["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity changed")
    if len(cache["ids"]) != protocol["expected_test_images"]:
        raise ValueError("Blind-test image count changed")
    return checkpoint, cache


@torch.inference_mode()
def evaluate_one(model, data, condition, recovery_steps=8, batch_size=24):
    target = data["ids"]
    labels = data["labels"]
    received_np, missing_np, physical = transmit_tokens(
        target.numpy(), 1024, condition["snr_esn0_db"], condition["channel"], condition["seed"], 4
    )
    received = torch.from_numpy(received_np).long()
    missing = torch.from_numpy(missing_np)
    restored = []
    for start in range(0, len(target), batch_size):
        restored.append(model.recover(
            received[start:start + batch_size].masked_fill(missing[start:start + batch_size], model.codes),
            missing[start:start + batch_size], steps=recovery_steps,
        ))
    restored = torch.cat(restored)
    spatial = torch.from_numpy(spatial_fill(received_np, missing_np)).long()
    missing_count = int(missing.sum())
    per_class = []
    for label in range(len(data["class_names"])):
        chosen = labels == label
        class_missing = missing[chosen]
        per_class.append({
            "label": label,
            "name": data["class_names"][label],
            "images": int(chosen.sum()),
            "all_token_accuracy": float((restored[chosen] == target[chosen]).float().mean()),
            "missing_token_accuracy": (float((restored[chosen][class_missing] == target[chosen][class_missing]).float().mean()) if int(class_missing.sum()) else None),
        })
    return {
        **condition,
        "images": len(target),
        "tokens": int(target.numel()),
        "missing_tokens": missing_count,
        "received_all_token_accuracy": float((((~missing) & (received == target)).sum()) / target.numel()),
        "recovered_all_token_accuracy": float((restored == target).float().mean()),
        "maskgit_missing_accuracy": (float((restored[missing] == target[missing]).float().mean()) if missing_count else None),
        "spatial_missing_accuracy": (float((spatial[missing] == target[missing]).float().mean()) if missing_count else None),
        "undetected_wrong_tokens": int(((~missing) & (received != target)).sum()),
        "per_class": per_class,
        "physical": physical,
        "test_accessed": True,
        "used_for_training": False,
    }


def summarize_groups(rows):
    def optional_mean(values):
        present = [value for value in values if value is not None]
        return float(np.mean(present)) if present else None

    summary = []
    for channel in ["awgn", "rayleigh"]:
        for snr in [0, 5, 10, 20]:
            chosen = [row for row in rows if row["channel"] == channel and row["snr_esn0_db"] == snr]
            if len(chosen) != 3:
                raise ValueError("Final summary requires exactly three seeds per channel point")
            summary.append({
                "channel": channel,
                "snr_esn0_db": snr,
                "seeds": 3,
                "measured_per_mean": float(np.mean([row["physical"]["measured_per"] for row in chosen])),
                "measured_per_std": float(np.std([row["physical"]["measured_per"] for row in chosen])),
                "payload_ber_mean": float(np.mean([row["physical"]["payload_ber"] for row in chosen])),
                "recovered_all_token_accuracy_mean": float(np.mean([row["recovered_all_token_accuracy"] for row in chosen])),
                "recovered_all_token_accuracy_std": float(np.std([row["recovered_all_token_accuracy"] for row in chosen])),
                "maskgit_missing_accuracy_mean": optional_mean([row["maskgit_missing_accuracy"] for row in chosen]),
                "spatial_missing_accuracy_mean": optional_mean([row["spatial_missing_accuracy"] for row in chosen]),
                "undetected_wrong_tokens": sum(row["undetected_wrong_tokens"] for row in chosen),
            })
    return summary


def summarize(rows, protocol):
    return {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "test_manifest_sha256": json.loads((ROOT / protocol["holdout_directory"] / "manifest.json").read_text())["test_manifest_sha256"],
        "test_images": protocol["expected_test_images"],
        "conditions": len(rows),
        "summary": summarize_groups(rows),
        "test_accessed": True,
        "used_for_training": False,
    }


def main() -> None:
    torch.set_num_threads(4)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checkpoint_path, data = validate(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    keys = [key(row) for row in rows]
    expected = {key(row) for row in protocol["conditions"]}
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or unplanned final-test result")
    pending = [row for row in protocol["conditions"] if key(row) not in set(keys)]
    if pending:
        model = load_context_maskgit_v2(torch.load(checkpoint_path, weights_only=True)).eval()
        row = evaluate_one(model, data, pending[0], protocol["recovery_steps"], protocol["evaluation_batch_size"])
        rows.append(row)
        atomic_json(results_path, rows)
        print(json.dumps({"completed_condition": key(row), "count": len(rows), "planned": len(expected)}))
    if len(rows) == len(expected):
        atomic_json(OUT / "summary.json", summarize(rows, protocol))
        atomic_json(OUT / "status.json", {"completed": True, "conditions": len(rows), "test_accessed": True, "used_for_training": False})


if __name__ == "__main__":
    main()
