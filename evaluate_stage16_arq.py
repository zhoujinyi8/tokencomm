"""Resumable Stage-16 comparison of semantic recovery, selective ARQ, and full ARQ."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from context_maskgit_v3 import load_context_maskgit_v3
from tokcom_arq import analytical_arq_metrics, transmit_tokens_with_arq


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE16_ARQ_TCE_PROTOCOL.json"
OUT = ROOT / "runs/stage16_arq_tce"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def key(row: dict) -> tuple:
    return row["channel"], float(row["snr_esn0_db"]), int(row["seed"]), row["policy_name"]


def validate(protocol: dict) -> tuple[torch.nn.Module, dict]:
    if protocol.get("status") != "frozen":
        raise ValueError("Stage-16 protocol must be frozen")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-16 implementation changed: {name}")
    checkpoint_path = ROOT / protocol["checkpoint"]
    cache_path = ROOT / protocol["validation_cache"]
    if sha256(checkpoint_path) != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen contextual checkpoint changed")
    if sha256(cache_path) != protocol["validation_cache_sha256"]:
        raise ValueError("Frozen development cache changed")
    status = json.loads((ROOT / "runs/stage15_contextual_tokcom/status.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status.get("test_accessed") or status.get("stage14_holdout_used"):
        raise ValueError("Stage-15 completion or isolation audit failed")
    data = torch.load(cache_path, weights_only=True)
    if len(data["ids"]) != protocol["images"]:
        raise ValueError("Frozen development image count changed")
    if data["identity"]["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity changed")
    model = load_context_maskgit_v3(torch.load(checkpoint_path, weights_only=True)).eval()
    return model, data


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for channel in ("awgn", "rayleigh"):
        snr = 5.0 if channel == "awgn" else 10.0
        for policy_name in ("semantic_only", "selective_001", "selective_002", "selective_004", "full_arq"):
            chosen = [
                row for row in rows
                if row["channel"] == channel
                and row["snr_esn0_db"] == snr
                and row["policy_name"] == policy_name
            ]
            if len(chosen) != 3:
                raise ValueError("Stage-16 summary requires exactly three seeds per point")
            metrics = [row["metrics"] for row in chosen]
            summary.append(
                {
                    "channel": channel,
                    "snr_esn0_db": snr,
                    "policy_name": policy_name,
                    "seeds": 3,
                    "initial_per_mean": float(np.mean([item["initial_per"] for item in metrics])),
                    "packet_attempts_mean": float(np.mean([item["packet_attempts"] for item in metrics])),
                    "retransmission_packets_mean": float(np.mean([item["retransmission_packets"] for item in metrics])),
                    "transmissions_per_packet_mean": float(np.mean([item["transmissions_per_packet"] for item in metrics])),
                    "token_error_rate_mean": float(np.mean([item["token_error_rate"] for item in metrics])),
                    "token_error_rate_std": float(np.std([item["token_error_rate"] for item in metrics])),
                    "recovered_token_accuracy_mean": float(np.mean([item["recovered_token_accuracy"] for item in metrics])),
                    "tce_mean": float(np.mean([item["tce_pixels_per_token_bit"] for item in metrics])),
                    "final_erasure_rate_mean": float(np.mean([item["final_erasure_rate"] for item in metrics])),
                    "undetected_error_packets": int(sum(item["undetected_error_packets"] for item in metrics)),
                }
            )
    return summary


def main() -> None:
    torch.set_num_threads(4)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    model, data = validate(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    rows = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
    keys = [key(row) for row in rows]
    expected = {key(row) for row in protocol["conditions"]}
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or unplanned Stage-16 result")
    pending = [row for row in protocol["conditions"] if key(row) not in set(keys)]
    if not pending:
        atomic_json(
            OUT / "summary.json",
            {
                "conditions": len(rows),
                "summary": summarize(rows),
                "test_accessed": False,
                "stage14_holdout_used": False,
            },
        )
        atomic_json(
            OUT / "status.json",
            {"completed": True, "conditions": len(rows), "test_accessed": False, "stage14_holdout_used": False},
        )
        print(json.dumps({"completed": True, "conditions": len(rows)}))
        return

    condition = pending[0]
    policy = protocol["policies"][condition["policy_name"]]
    restored, missing, metrics = transmit_tokens_with_arq(
        data["ids"].numpy(),
        receiver=model,
        labels=data["labels"].long(),
        codebook_size=1024,
        snr_db=float(condition["snr_esn0_db"]),
        channel=condition["channel"],
        seed=int(condition["seed"]),
        packet_tokens=4,
        max_rounds=int(protocol["max_rounds"]),
        policy=policy["implementation_policy"],
        confidence_threshold=float(policy["confidence_threshold"]),
        recovery_steps=int(protocol["recovery_steps"]),
        guidance_scale=float(protocol["guidance_scale"]),
    )
    initial = metrics["rounds"][0]
    initial_per = initial["crc_failures"] / initial["attempted_packets"]
    analytical = analytical_arq_metrics(initial_per)
    metrics.update(
        {
            "initial_per": initial_per,
            "paper_unbounded_arq_expected_transmissions": analytical["expected_transmissions_per_packet"],
            "paper_unbounded_arq_tce": analytical["tce_pixels_per_token_bit"],
            "confidence_threshold": float(policy["confidence_threshold"]),
        }
    )
    row = {
        **condition,
        "images": len(data["ids"]),
        "tokens": int(data["ids"].numel()),
        "metrics": metrics,
        "restored_matches_shape": list(restored.shape) == list(data["ids"].shape),
        "remaining_erasure_count": int(missing.sum()),
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    rows.append(row)
    atomic_json(result_path, rows)
    print(json.dumps({"completed_condition": key(row), "count": len(rows), "planned": len(expected)}))
    if len(rows) == len(expected):
        atomic_json(
            OUT / "summary.json",
            {
                "conditions": len(rows),
                "summary": summarize(rows),
                "test_accessed": False,
                "stage14_holdout_used": False,
            },
        )
        atomic_json(
            OUT / "status.json",
            {"completed": True, "conditions": len(rows), "test_accessed": False, "stage14_holdout_used": False},
        )


if __name__ == "__main__":
    main()
