import json
import unittest
import hashlib
import tempfile
from pathlib import Path

import torch

from context_maskgit_v4 import load_context_maskgit_v4
from evaluate_stage19_v4_link import condition_key
from snapshot_stage19 import paired_deltas


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

    def test_frozen_implementation_and_assets_match(self):
        protocol = json.loads((ROOT / "STAGE19_V4_LINK_PROTOCOL.json").read_text(encoding="utf-8"))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, expected in protocol["implementation_sha256"].items():
            self.assertEqual(digest(ROOT / name), expected, name)
        for field, hash_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("validation_cache", "validation_cache_sha256"),
            ("clip_checkpoint", "clip_checkpoint_sha256"),
            ("vqgan_checkpoint", "vqgan_checkpoint_sha256"),
        ):
            self.assertEqual(digest(ROOT / protocol[field]), protocol[hash_field], field)

    def test_hidden_dispatch_is_bounded_and_single_writer(self):
        dispatch = (ROOT / "dispatch_stage19.ps1").read_text(encoding="utf-8")
        wrapper = (ROOT / "dispatch_stage19_hidden.vbs").read_text(encoding="utf-8")
        self.assertIn("evaluate_stage19_v4_link\\.py", dispatch)
        self.assertIn("--max-seconds', '900", dispatch)
        self.assertIn("-WindowStyle Hidden", dispatch)
        self.assertIn("shell.Run command, 0, False", wrapper)

    def test_live_delta_requires_image_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old" / "per_image"
            new = root / "new" / "per_image"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            fields = {"token_accuracy": 0.5, "psnr": 10.0, "clip_cosine": 0.6, "lpips": 0.4, "vqgan_baseline_psnr": 20.0}
            (old / "0000.json").write_text(json.dumps({"index": 0, "sha256": "abc", **fields}))
            changed = dict(fields, token_accuracy=0.6, psnr=11.0, clip_cosine=0.7, lpips=0.3)
            (new / "0000.json").write_text(json.dumps({"index": 0, "sha256": "abc", **changed}))
            result = paired_deltas(root / "old", root / "new")
            self.assertEqual(result["paired_images"], 1)
            self.assertAlmostEqual(result["token_accuracy_delta_mean"], 0.1)
            self.assertAlmostEqual(result["lpips_delta_mean"], -0.1)


if __name__ == "__main__":
    unittest.main()
