import unittest

from evaluate_stage12_final import condition_key, planned_conditions, summarize


PROTOCOL = {
    "channels": ["awgn", "rayleigh"],
    "snr_db": [0, 10],
    "seeds": [11, 22],
    "controls": [
        {"variant": "wrong_class", "channel": "awgn", "snr_db": 10, "seed": 33},
        {"variant": "zero", "channel": "awgn", "snr_db": 10, "seed": 33},
    ],
}


def result(condition, accuracy=0.8, loss=0.2):
    return {
        **condition,
        "count": 20,
        "loss": loss,
        "exact_match": accuracy,
        "per_class_accuracy": {"0": accuracy, "1": accuracy},
        "error_indices": [],
    }


class Stage12FinalTests(unittest.TestCase):
    def test_plan_is_unique_and_has_all_controls(self):
        plan = planned_conditions(PROTOCOL)
        self.assertEqual(len(plan), 10)
        self.assertEqual(len({condition_key(row) for row in plan}), 10)

    def test_complete_summary(self):
        rows = [result(row) for row in planned_conditions(PROTOCOL)]
        summary, controls, complete = summarize(rows, PROTOCOL)
        self.assertTrue(complete)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(row["seeds"] == 2 for row in summary))

    def test_partial_summary_is_not_complete(self):
        rows = [result(row) for row in planned_conditions(PROTOCOL)[:-1]]
        _, _, complete = summarize(rows, PROTOCOL)
        self.assertFalse(complete)

    def test_unknown_or_duplicate_results_fail_closed(self):
        plan = planned_conditions(PROTOCOL)
        duplicate = [result(plan[0]), result(plan[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(duplicate, PROTOCOL)
        unknown = result({"variant": "correct", "kind": "awgn", "snr_db": 99.0, "seed": 11})
        with self.assertRaisesRegex(ValueError, "not present"):
            summarize([unknown], PROTOCOL)


if __name__ == "__main__":
    unittest.main()
