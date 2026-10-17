import json
import tempfile
import unittest
from pathlib import Path

import torch

from context_maskgit_v4 import ContextMaskGITv4
from context_maskgit_v5 import initialize_context_maskgit_v5_from_v4
from train_stage22_contextual_tokcom import ROOT, balanced_indices, evaluate, read_history, sha256_file


class Stage22ContextualTokComTests(unittest.TestCase):
    def test_training_plan_is_pre_registered_and_matches_implementation(self):
        plan = json.loads((ROOT / "STAGE22_TRAINING_PLAN.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "pre_registered")
        self.assertEqual(plan["training"]["steps"], 12000)
        self.assertEqual(plan["training"]["effective_sample_draws"], 384000)
        self.assertEqual(plan["validation"]["images"], 1010)
        self.assertFalse(plan["test_accessed"])
        self.assertFalse(plan["test_pixels_extracted"])
        for name, expected in plan["implementation_sha256"].items():
            self.assertEqual(sha256_file(ROOT / name), expected)

    def test_balanced_subset_is_deterministic(self):
        rows = [
            {"image": f"images/{label}/{index}.jpg", "label": label}
            for label in range(101)
            for index in range(3)
        ]
        first = balanced_indices(rows, 2, 123)
        second = balanced_indices(list(rows), 2, 123)
        self.assertTrue(torch.equal(first, second))
        labels = torch.tensor([rows[index]["label"] for index in first])
        self.assertTrue(torch.equal(torch.bincount(labels, minlength=101), torch.full((101,), 2)))

    def test_history_recovery_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text(json.dumps({"step": 1}) + "\n" + json.dumps({"step": 3}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_history(path, 3)

    def test_validation_reports_all_context_controls(self):
        torch.manual_seed(22)
        base = ContextMaskGITv4(d=16, layers=1, local_blocks=1, classes=10).eval()
        checkpoint = {"model": "context_maskgit_v4", "config": base.config, "state": base.state_dict()}
        model = initialize_context_maskgit_v5_from_v4(
            checkpoint,
            torch.randn(101, 8),
            [f"class_{index}" for index in range(101)],
        ).eval()
        protocol = {
            "training": {"packet_tokens": 4},
            "validation": {"per": 0.3, "batch_size": 2},
        }
        rows = evaluate(model, torch.randint(0, 1024, (4, 256)), torch.tensor([0, 1, 2, 3]), protocol, 2201)
        self.assertEqual([row["mode"] for row in rows], ["correct", "wrong", "none"])
        self.assertTrue(all(row["missing_tokens"] > 0 for row in rows))
        self.assertTrue(all(torch.isfinite(torch.tensor(row["masked_ce"])) for row in rows))


if __name__ == "__main__":
    unittest.main()
