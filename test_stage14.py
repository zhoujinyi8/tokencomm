import unittest

import torch

from context_maskgit_v2 import ContextMaskGITv2
from evaluate_stage14_final import evaluate_one, summarize_groups
from freeze_stage14_protocol import conditions


class Stage14Tests(unittest.TestCase):
    def test_blind_plan_is_unique_and_matches_shared_grid(self):
        rows = conditions()
        keys = {(row["channel"], row["snr_esn0_db"], row["seed"]) for row in rows}
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(keys), 24)
        self.assertEqual({row["channel"] for row in rows}, {"awgn", "rayleigh"})
        self.assertEqual({row["snr_esn0_db"] for row in rows}, {0.0, 5.0, 10.0, 20.0})

    def test_noiseless_final_evaluator_preserves_tokens_and_reports_classes(self):
        model = ContextMaskGITv2(d=16, layers=1, local_blocks=1).eval()
        target = torch.randint(0, 1024, (4, 256))
        data = {"ids": target, "labels": torch.tensor([0, 0, 1, 1]), "class_names": ["a", "b"]}
        row = evaluate_one(
            model, data, {"channel": "noiseless", "snr_esn0_db": 100.0, "seed": 14},
            recovery_steps=2, batch_size=3,
        )
        self.assertEqual(row["recovered_all_token_accuracy"], 1.0)
        self.assertEqual(row["missing_tokens"], 0)
        self.assertEqual([entry["images"] for entry in row["per_class"]], [2, 2])
        self.assertTrue(row["test_accessed"])
        self.assertFalse(row["used_for_training"])

    def test_summary_requires_three_seeds_and_handles_zero_loss_points(self):
        rows = []
        for condition in conditions():
            no_loss = condition["snr_esn0_db"] == 20
            rows.append({
                **condition,
                "recovered_all_token_accuracy": 1.0,
                "maskgit_missing_accuracy": None if no_loss else 0.5,
                "spatial_missing_accuracy": None if no_loss else 0.25,
                "undetected_wrong_tokens": 0,
                "physical": {"measured_per": 0.0 if no_loss else 0.2, "payload_ber": 0.0},
            })
        groups = summarize_groups(rows)
        self.assertEqual(len(groups), 8)
        high_snr = [row for row in groups if row["snr_esn0_db"] == 20]
        self.assertTrue(all(row["maskgit_missing_accuracy_mean"] is None for row in high_snr))
        with self.assertRaises(ValueError):
            summarize_groups(rows[:-1])

    def test_dispatcher_orders_freeze_before_test_access(self):
        from evaluate_stage14_final import ROOT

        source = (ROOT / "dispatch_stage13.ps1").read_text(encoding="utf-8")
        freeze = source.index("freeze_stage14_protocol.py")
        prepare = source.index("prepare_stage14_holdout.py")
        evaluate = source.index("evaluate_stage14_final.py")
        self.assertLess(freeze, prepare)
        self.assertLess(prepare, evaluate)
        self.assertIn("STAGE14_TOKCOM_FINAL_PROTOCOL.json", source)


if __name__ == "__main__":
    unittest.main()
