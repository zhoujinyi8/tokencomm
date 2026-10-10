import unittest

from finalize_stage18 import aggregate_conditions


class FinalizeStage18Tests(unittest.TestCase):
    def test_condition_aggregation_keeps_controls_separate(self):
        entries = [
            {"conditions": [
                {"mode": "correct", "ce": 1.0, "accuracy": 0.5},
                {"mode": "wrong", "ce": 2.0, "accuracy": 0.2},
                {"mode": "none", "ce": 1.5, "accuracy": 0.3},
            ]},
            {"conditions": [
                {"mode": "correct", "ce": 1.2, "accuracy": 0.4},
                {"mode": "wrong", "ce": 2.2, "accuracy": 0.1},
                {"mode": "none", "ce": 1.7, "accuracy": 0.2},
            ]},
        ]
        result = aggregate_conditions(entries)
        self.assertAlmostEqual(result["correct"]["ce_mean"], 1.1)
        self.assertAlmostEqual(result["wrong"]["accuracy_mean"], 0.15)
        self.assertEqual(result["none"]["seeds"], 2)


if __name__ == "__main__":
    unittest.main()
