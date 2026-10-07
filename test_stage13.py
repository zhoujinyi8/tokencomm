import json
import tempfile
import unittest
from pathlib import Path

import torch

from train_stage13_tokcom import ROOT, build_model, read_history, sha256
from report_stage13 import load_completed


class Stage13Tests(unittest.TestCase):
    def test_frozen_data_hashes_and_budget(self):
        protocol = json.loads((ROOT / "STAGE13_TOKCOM_PROTOCOL.json").read_text(encoding="utf-8"))
        self.assertEqual(sha256(ROOT / "data/imagenette_stage10/manifest.json"), protocol["data"]["manifest_sha256"])
        self.assertEqual(sha256(ROOT / "data/imagenette_stage10/train.jsonl"), protocol["data"]["train_jsonl_sha256"])
        self.assertEqual(protocol["training"]["steps"], 6000)
        self.assertEqual(protocol["data"]["training_images"], 8000)
        self.assertFalse(protocol["test_accessed"])

    def test_history_recovery_deduplicates_and_truncates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            rows = [{"step": 1, "loss": 2}, {"step": 2, "loss": 1}, {"step": 2, "loss": 1}, {"step": 3, "loss": 0}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            kept = read_history(path, 2)
            self.assertEqual([row["step"] for row in kept], [1, 2])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_frozen_seed_controls_initial_weights(self):
        protocol = json.loads((ROOT / "STAGE13_TOKCOM_PROTOCOL.json").read_text(encoding="utf-8"))
        first = build_model(protocol)
        torch.rand(7)
        second = build_model(protocol)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values())))

    def test_report_refuses_incomplete_run(self):
        if not (ROOT / "runs/stage13_tokcom_large/status.json").exists():
            with self.assertRaises(FileNotFoundError):
                load_completed()

    def test_cache_cli_exposes_bounded_chunk_mode(self):
        source = (ROOT / "cache_real_tokens.py").read_text(encoding="utf-8")
        self.assertIn("--max-seconds", source)
        self.assertIn("SystemExit(75)", source)
        self.assertIn("aggregate already complete", source)

    def test_trainer_exposes_bounded_recovery_mode(self):
        source = (ROOT / "train_stage13_tokcom.py").read_text(encoding="utf-8")
        self.assertIn('"--max-seconds"', source)
        self.assertIn('"TRAIN_CHUNK_COMPLETE"', source)
        self.assertIn("raise SystemExit(75)", source)


if __name__ == "__main__":
    unittest.main()
