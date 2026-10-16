import json
import unittest
from pathlib import Path

import torch

from context_maskgit_v4 import load_context_maskgit_v4
from evaluate_stage19_v4_link import condition_key


ROOT = Path(__file__).resolve().parent


class Stage19Tests(unittest.TestCase):
    def test_v4_checkpoint_and_condition_grid(self):
        checkpoint = torch.load(ROOT / "runs/stage18_strong_cmi/best.pt", weights_only=True)
        model = load_context_maskgit_v4(checkpoint)
        self.assertEqual(model.codes, 1024)
        protocol = json.loads((ROOT / "STAGE19_V4_LINK_PROTOCOL.json").read_text(encoding="utf-8"))
        keys = [condition_key(row) for row in protocol["conditions"]]
        self.assertEqual(len(keys), 24)
        self.assertEqual(len(set(keys)), 24)
        self.assertFalse(protocol["test_accessed"])
        self.assertFalse(protocol["stage14_holdout_used"])


if __name__ == "__main__":
    unittest.main()
