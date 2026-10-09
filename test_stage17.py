import unittest
from pathlib import Path
import json
import tempfile

import numpy as np

from evaluate_stage17_perceptual import atomic_json, condition_key
from freeze_stage17_perceptual_protocol import choose_selective


class Stage17Tests(unittest.TestCase):
    def test_condition_key_is_complete(self):
        row = {"channel": "awgn", "snr_esn0_db": 5.0, "seed": 3, "mode": "no_cmi"}
        self.assertEqual(condition_key(row), "awgn_5p0db_seed3_no_cmi")

    def test_selective_rule_prefers_balanced_point(self):
        rows = []
        for channel in ("awgn", "rayleigh"):
            rows.extend([
                {"channel": channel, "policy_name": "semantic_only", "recovered_token_accuracy_mean": 0.5, "tce_mean": 1.0},
                {"channel": channel, "policy_name": "selective_001", "recovered_token_accuracy_mean": 0.55, "tce_mean": 0.95},
                {"channel": channel, "policy_name": "selective_002", "recovered_token_accuracy_mean": 0.75, "tce_mean": 0.75},
                {"channel": channel, "policy_name": "selective_004", "recovered_token_accuracy_mean": 0.95, "tce_mean": 0.55},
                {"channel": channel, "policy_name": "full_arq", "recovered_token_accuracy_mean": 1.0, "tce_mean": 0.5},
            ])
        selected, scores = choose_selective(rows)
        self.assertEqual(selected, "selective_002")
        self.assertGreater(scores[selected]["mean_harmonic"], scores["selective_001"]["mean_harmonic"])

    def test_hidden_dispatch_is_single_writer_and_bounded(self):
        root = Path(__file__).resolve().parent
        dispatch = (root / "dispatch_stage17.ps1").read_text(encoding="utf-8")
        wrapper = (root / "dispatch_stage17_hidden.vbs").read_text(encoding="utf-8")
        self.assertIn("evaluate_stage17_perceptual\\.py", dispatch)
        self.assertIn("--max-seconds', '220", dispatch)
        self.assertIn("-WindowStyle Hidden", dispatch)
        self.assertIn("shell.Run command, 0, False", wrapper)

    def test_atomic_json_converts_numpy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            atomic_json(path, {"scalar": np.float64(0.25), "array": np.array([1, 2])})
            self.assertEqual(json.loads(path.read_text()), {"scalar": 0.25, "array": [1, 2]})


if __name__ == "__main__":
    unittest.main()
