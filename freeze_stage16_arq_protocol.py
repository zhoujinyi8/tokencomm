"""Freeze the Stage-16 ARQ/TCE development comparison."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE16_ARQ_TCE_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    executable = shutil.which("git")
    if executable is None:
        executable = next(
            (str(path) for path in [Path("C:/Program Files/Git/cmd/git.exe"), Path("C:/Program Files/Git/bin/git.exe")] if path.exists()),
            None,
        )
    if executable is None:
        raise FileNotFoundError("git.exe is required")
    return subprocess.run([executable, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    if OUTPUT.exists():
        print(json.dumps({"already_frozen": True, "sha256": sha256(OUTPUT)}))
        return
    status = json.loads((ROOT / "runs/stage15_contextual_tokcom/status.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "runs/stage15_contextual_tokcom/results.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status.get("test_accessed") or status.get("stage14_holdout_used"):
        raise ValueError("Stage-15 training is incomplete or contaminated")
    if int(results["best_step"]) != 2000:
        raise ValueError("Unexpected Stage-15 best checkpoint")
    guidance_rows = [row for row in results["guidance_sweep"] if row["guidance_scale"] == 1.0]
    if len(guidance_rows) != 3:
        raise ValueError("Frozen guidance scale does not have three validation seeds")
    checkpoint = ROOT / "runs/stage15_contextual_tokcom/best.pt"
    cache = ROOT / "data/imagenette_stage10/token_cache/val.pt"
    cache_data = torch.load(cache, weights_only=True)
    policies = {
        "semantic_only": {"implementation_policy": "confidence", "confidence_threshold": 0.0},
        "selective_001": {"implementation_policy": "confidence", "confidence_threshold": 0.01},
        "selective_002": {"implementation_policy": "confidence", "confidence_threshold": 0.02},
        "selective_004": {"implementation_policy": "confidence", "confidence_threshold": 0.04},
        "full_arq": {"implementation_policy": "all_failed", "confidence_threshold": 1.0},
    }
    conditions = [
        {"channel": channel, "snr_esn0_db": snr, "seed": seed, "policy_name": policy}
        for channel, snr in [("awgn", 5.0), ("rayleigh", 10.0)]
        for seed in [16101, 16102, 16103]
        for policy in policies
    ]
    implementation = ["evaluate_stage16_arq.py", "tokcom_arq.py", "context_maskgit_v3.py", "digital_link.py"]
    payload = {
        "stage": 16,
        "purpose": "Compare semantic recovery, confidence-selective ARQ, and conventional CRC-ARQ",
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_commit": git_head(),
        "implementation_sha256": {name: sha256(ROOT / name) for name in implementation},
        "checkpoint": "runs/stage15_contextual_tokcom/best.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "stage15_results_sha256": sha256(ROOT / "runs/stage15_contextual_tokcom/results.json"),
        "validation_cache": "data/imagenette_stage10/token_cache/val.pt",
        "validation_cache_sha256": sha256(cache),
        "tokenizer_sha256": cache_data["identity"]["tokenizer_sha256"],
        "images": len(cache_data["ids"]),
        "guidance_scale": 1.0,
        "guidance_selection": "lowest correct-context development CE and highest exact recovery in frozen Stage-15 sweep",
        "confidence_calibration": {
            "split": "Stage-15 development only",
            "token_max_probability_median": 0.01478590164333582,
            "thresholds": [0.01, 0.02, 0.04],
        },
        "max_rounds": 4,
        "recovery_steps": 8,
        "packet_tokens": 4,
        "policies": policies,
        "conditions": conditions,
        "metrics": [
            "initial_per",
            "packet_attempts",
            "retransmission_packets",
            "transmissions_per_packet",
            "token_error_rate",
            "recovered_token_accuracy",
            "final_erasure_rate",
            "tce_pixels_per_token_bit",
            "paper_unbounded_arq_expected_transmissions",
            "paper_unbounded_arq_tce",
            "undetected_error_packets",
        ],
        "scope": "development comparison; no test or Stage-14 holdout access",
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "conditions": len(conditions), "sha256": sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
