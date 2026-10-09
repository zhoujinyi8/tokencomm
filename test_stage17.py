import unittest

from evaluate_stage17_perceptual import condition_key
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


if __name__ == "__main__":
    unittest.main()
