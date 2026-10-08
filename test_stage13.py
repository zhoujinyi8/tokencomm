import json
import tempfile
import unittest
from pathlib import Path

import torch

from train_stage13_tokcom import ROOT, build_model, read_history, sha256
from report_stage13 import load_completed
from evaluate_stage13_physical import evaluate_one, key, plan


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

    def test_dispatcher_uses_hidden_single_writer_workers(self):
        source = (ROOT / "dispatch_stage13.ps1").read_text(encoding="utf-8")
        self.assertIn("-WindowStyle Hidden", source)
        self.assertIn("Worker already active", source)
        self.assertIn("--max-seconds", source)
        self.assertIn("report_fair_comparison.py", source)
        self.assertIn("TOKCOM_TAICHI_FAIR_COMPARISON.md", source)

    def test_physical_plan_is_unique_and_development_only(self):
        rows = plan()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({key(row) for row in rows}), 24)
        self.assertEqual({row["channel"] for row in rows}, {"awgn", "rayleigh"})
        self.assertEqual({row["snr_esn0_db"] for row in rows}, {0.0, 5.0, 10.0, 20.0})

    def test_fair_comparison_grid_validation_fails_closed(self):
        from report_fair_comparison import validate_shared_grid

        tok = [
            {"channel": channel, "snr_esn0_db": float(snr), "seeds": 3}
            for channel in ["awgn", "rayleigh"] for snr in [0, 5, 10, 20]
        ]
        tai = [
            {"kind": channel, "snr_db": float(snr), "seeds": 3}
            for channel in ["awgn", "rayleigh"] for snr in [0, 5, 10, 20]
        ]
        tok_map, tai_map = validate_shared_grid(tok, tai)
        self.assertEqual(len(tok_map), 8)
        self.assertEqual(len(tai_map), 8)
        with self.assertRaises(ValueError):
            validate_shared_grid(tok[:-1], tai)
        bad_seeds = [dict(row) for row in tai]
        bad_seeds[0]["seeds"] = 2
        with self.assertRaises(ValueError):
            validate_shared_grid(tok, bad_seeds)

    def test_noiseless_physical_path_preserves_all_tokens(self):
        from context_maskgit_v2 import ContextMaskGITv2

        model = ContextMaskGITv2(d=16, layers=1, local_blocks=1).eval()
        data = {"ids": torch.randint(0, 1024, (2, 256))}
        row = evaluate_one(model, data, {"channel": "noiseless", "snr_esn0_db": 100.0, "seed": 4})
        self.assertEqual(row["missing_tokens"], 0)
        self.assertEqual(row["received_all_token_accuracy"], 1.0)
        self.assertEqual(row["recovered_all_token_accuracy"], 1.0)
        self.assertIsNone(row["maskgit_missing_accuracy"])


if __name__ == "__main__":
    unittest.main()
