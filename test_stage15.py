import unittest

import torch

from context_maskgit_v3 import ContextMaskGITv3
from train_stage15_contextual_tokcom import evaluate


class Stage15Tests(unittest.TestCase):
    def test_evaluate_uses_identical_masks_for_context_controls(self):
        torch.set_num_threads(2)
        torch.manual_seed(1515)
        model = ContextMaskGITv3(d=16, layers=1, local_blocks=1, classes=10).eval()
        data = {
            "ids": torch.randint(0, 1024, (4, 256)),
            "labels": torch.arange(4),
        }
        rows = evaluate(model, data, per=0.3, seed=91, recovery_steps=2, batch_size=4)
        self.assertEqual([row["mode"] for row in rows], ["correct", "wrong", "none"])
        self.assertEqual(len({row["missing_tokens"] for row in rows}), 1)
        self.assertTrue(all(row["missing_tokens"] > 0 for row in rows))

    def test_evaluate_is_deterministic_for_fixed_seed(self):
        torch.set_num_threads(2)
        torch.manual_seed(1516)
        model = ContextMaskGITv3(d=16, layers=1, local_blocks=1, classes=10).eval()
        data = {"ids": torch.randint(0, 1024, (3, 256)), "labels": torch.arange(3)}
        first = evaluate(model, data, 0.5, 123, recovery_steps=2, batch_size=3)
        second = evaluate(model, data, 0.5, 123, recovery_steps=2, batch_size=3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
