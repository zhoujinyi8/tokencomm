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


if __name__ == "__main__":
    unittest.main()
