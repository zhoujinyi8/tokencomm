import unittest

from evaluate_stage16_arq import key, summarize


class Stage16Tests(unittest.TestCase):
    def test_key_includes_policy(self):
        row = {"channel": "awgn", "snr_esn0_db": 5, "seed": 1, "policy_name": "full_arq"}
        self.assertEqual(key(row), ("awgn", 5.0, 1, "full_arq"))

    def test_summary_requires_three_seeds_per_policy(self):
        rows = []
        policies = ("semantic_only", "selective_001", "selective_002", "selective_004", "full_arq")
        for channel, snr in (("awgn", 5.0), ("rayleigh", 10.0)):
            for policy in policies:
                for seed in (1, 2, 3):
                    rows.append(
                        {
                            "channel": channel,
                            "snr_esn0_db": snr,
                            "seed": seed,
                            "policy_name": policy,
                            "metrics": {
                                "initial_per": 0.2,
                                "packet_attempts": 120,
                                "retransmission_packets": 20,
                                "transmissions_per_packet": 1.2,
                                "token_error_rate": 0.1,
                                "recovered_token_accuracy": 0.9,
                                "tce_pixels_per_token_bit": 20.0,
                                "final_erasure_rate": 0.05,
                                "undetected_error_packets": 0,
                            },
                        }
                    )
        self.assertEqual(len(summarize(rows)), 10)
        with self.assertRaises(ValueError):
            summarize(rows[:-1])


if __name__ == "__main__":
    unittest.main()
