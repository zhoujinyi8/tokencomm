import unittest

from finalize_stage22 import aggregate_seed_conditions, audit


class FinalizeStage22Tests(unittest.TestCase):
    def test_condition_aggregation(self):
        entries = [
            {
                "seed": seed,
                "conditions": [
                    {"mode": mode, "masked_ce": float(index + seed), "one_pass_accuracy": 0.01 * index}
                    for index, mode in enumerate(("correct", "wrong", "none"))
                ],
            }
            for seed in (1, 2, 3)
        ]
        output = aggregate_seed_conditions(entries)
        self.assertEqual(set(output), {"correct", "wrong", "none"})
        self.assertEqual(output["correct"]["seeds"], 3)
        self.assertAlmostEqual(output["correct"]["masked_ce_mean"], 2.0)

    def test_audit_is_fail_closed_before_training(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            audit()


if __name__ == "__main__":
    unittest.main()
