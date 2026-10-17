import unittest
import json
from pathlib import Path

from stage23_link_common import balanced_window_indices, select_threshold


class Stage23LinkTests(unittest.TestCase):
    def test_pre_registered_real_windows_are_balanced_and_disjoint(self):
        root = Path(__file__).resolve().parent
        records_path = root / "data/stage20_food101/dev.jsonl"
        if not records_path.exists():
            self.skipTest("Food-101 development manifest is unavailable")
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        training_validation = balanced_window_indices(records, start_per_class=0, images_per_class=10, seed=2026102201)
        calibration = balanced_window_indices(records, start_per_class=10, images_per_class=5, seed=2026102201)
        evaluation = balanced_window_indices(records, start_per_class=15, images_per_class=5, seed=2026102201)
        self.assertEqual((len(training_validation), len(calibration), len(evaluation)), (1010, 505, 505))
        self.assertTrue(set(training_validation).isdisjoint(calibration))
        self.assertTrue(set(training_validation).isdisjoint(evaluation))
        self.assertTrue(set(calibration).isdisjoint(evaluation))

    def test_balanced_windows_are_disjoint(self):
        records = [
            {"label": label, "image": f"class{label}/{index}.jpg"}
            for label in range(3) for index in range(12)
        ]
        first = balanced_window_indices(records, start_per_class=0, images_per_class=4, seed=7, classes=3)
        second = balanced_window_indices(records, start_per_class=4, images_per_class=3, seed=7, classes=3)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(second), 9)
        self.assertTrue(set(first).isdisjoint(second))

    def test_selection_uses_efficiency_after_quality_constraint(self):
        rows = []
        values = {
            "semantic_only": (0.40, 25.6),
            "full_arq": (0.04, 15.0),
            "selective_0.01": (0.20, 24.0),
            "selective_0.02": (0.12, 21.0),
            "selective_0.04": (0.08, 18.0),
        }
        for mode, (error, tce) in values.items():
            for seed in (1, 2, 3):
                rows.append({"mode": mode, "seed": seed, "metrics": {"token_error_rate": error, "tce_pixels_per_token_bit": tce}})
        selected = select_threshold(rows, [0.01, 0.02, 0.04], 0.75)
        self.assertEqual(selected["threshold"], 0.02)
        self.assertFalse(selected["fallback_used"])

    def test_selection_fallback_is_deterministic(self):
        rows = []
        values = {"semantic_only": (0.40, 25.6), "full_arq": (0.0, 15.0), "selective_0.01": (0.30, 24.0), "selective_0.02": (0.20, 20.0)}
        for mode, (error, tce) in values.items():
            rows.append({"mode": mode, "metrics": {"token_error_rate": error, "tce_pixels_per_token_bit": tce}})
        selected = select_threshold(rows, [0.01, 0.02], 0.75)
        self.assertEqual(selected["threshold"], 0.02)
        self.assertTrue(selected["fallback_used"])


if __name__ == "__main__":
    unittest.main()
