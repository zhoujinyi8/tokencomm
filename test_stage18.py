import unittest
from pathlib import Path


class Stage18Tests(unittest.TestCase):
    def test_training_is_resumable_and_excludes_holdouts(self):
        root = Path(__file__).resolve().parent
        source = (root / "train_stage18_contextual_tokcom.py").read_text(encoding="utf-8")
        self.assertIn('recovery["protocol_sha256"]', source)
        self.assertIn('optimizer.load_state_dict', source)
        self.assertIn('scheduler.load_state_dict', source)
        self.assertIn('torch.random.set_rng_state', source)
        self.assertNotIn("stage14_tokcom_final_holdout", source)

    def test_hidden_dispatch_is_single_writer_and_bounded(self):
        root = Path(__file__).resolve().parent
        dispatch = (root / "dispatch_stage18.ps1").read_text(encoding="utf-8")
        wrapper = (root / "dispatch_stage18_hidden.vbs").read_text(encoding="utf-8")
        self.assertIn("train_stage18_contextual_tokcom\\.py", dispatch)
        self.assertIn("--max-seconds', '220", dispatch)
        self.assertIn("-WindowStyle Hidden", dispatch)
        self.assertIn("shell.Run command, 0, False", wrapper)


if __name__ == "__main__":
    unittest.main()
