"""Pre-register the Stage-23 V5 communication-link evaluation before Stage-22 results."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from train_stage22_contextual_tokcom import sha256_file


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE23_V5_LINK_PLAN.json"


def main() -> None:
    if OUTPUT.exists():
        print(json.dumps({"already_pre_registered": True, "sha256": sha256_file(OUTPUT)}))
        return
    stage22_plan = ROOT / "STAGE22_TRAINING_PLAN.json"
    if not stage22_plan.exists():
        raise FileNotFoundError("Stage-22 pre-registered training plan is required")
    payload = {
        "stage": 23,
        "status": "pre_registered",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Calibrate and evaluate V5 confidence-selective ARQ on held development tokens",
        "evidence_level": "Food-101 development link evaluation; not the sealed official test",
        "stage22_training_plan_sha256": sha256_file(stage22_plan),
        "data": {
            "dataset": "Food-101 development split",
            "ranking_seed": 2026102201,
            "stage22_model_selection_window_per_class": [0, 10],
            "threshold_calibration_window_per_class": [10, 15],
            "link_evaluation_window_per_class": [15, 20],
            "threshold_calibration_images": 505,
            "link_evaluation_images": 505,
            "classes": 101,
        },
        "link": {
            "channels": [
                {"channel": "awgn", "snr_esn0_db": 5.0},
                {"channel": "rayleigh", "snr_esn0_db": 10.0},
            ],
            "seeds": [23101, 23102, 23103],
            "packet_tokens": 4,
            "max_rounds": 4,
            "recovery_steps": 8,
            "guidance_scale": 1.0,
            "threshold_candidates": [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
            "calibration_anchors": ["semantic_only", "full_arq"],
            "evaluation_modes": ["no_cmi", "cmi_no_arq", "cmi_selective", "full_arq"],
        },
        "threshold_selection": {
            "per_channel": True,
            "target_recovery_fraction": 0.75,
            "rule": "maximum mean TCE among thresholds closing at least 75% of the calibration token-error gap from semantic-only to full ARQ; ties use lower token error then lower threshold; if none qualify, minimum token error then maximum TCE",
            "evaluation_subset_used_for_selection": False,
        },
        "metrics": [
            "initial_per", "packet_attempts", "retransmission_packets", "transmissions_per_packet",
            "token_error_rate", "recovered_token_accuracy", "final_erasure_rate",
            "tce_pixels_per_token_bit", "undetected_error_packets",
        ],
        "official_test_images_sealed": 25250,
        "test_pixels_extracted": False,
        "test_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pre_registered": True, "sha256": sha256_file(OUTPUT)}))


if __name__ == "__main__":
    main()
