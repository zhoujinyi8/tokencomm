import unittest
import hashlib
import json
from pathlib import Path

from prepare_stage20_food101 import safe_name, split_train


class Stage20Food101Tests(unittest.TestCase):
    def test_safe_archive_paths(self):
        self.assertEqual(str(safe_name("food-101/images/a/b.jpg")), "food-101/images/a/b.jpg")
        for name in ("../escape", "/absolute", "food-101/../escape"):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_split_is_deterministic_and_balanced(self):
        classes = ["a", "b"]
        rows = [f"{name}/{index}" for name in classes for index in range(750)]
        first = split_train(rows, classes, 50, 123)
        second = split_train(list(reversed(rows)), classes, 50, 123)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 1400)
        self.assertEqual(len(first[1]), 100)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_frozen_plan_and_hidden_resumable_dispatch(self):
        root = Path(__file__).resolve().parent
        plan = json.loads((root / "STAGE20_FOOD101_SCALE_PLAN.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256((root / "prepare_stage20_food101.py").read_bytes()).hexdigest()
        self.assertEqual(plan["implementation_sha256"]["prepare_stage20_food101.py"], expected)
        self.assertFalse(plan["test_accessed"])
        dispatch = (root / "dispatch_stage20_data.ps1").read_text(encoding="utf-8")
        wrapper = (root / "dispatch_stage20_data_hidden.vbs").read_text(encoding="utf-8")
        self.assertIn("prepare_stage20_food101\\.py", dispatch)
        self.assertIn("-WindowStyle Hidden", dispatch)
        self.assertIn("shell.Run command, 0, False", wrapper)


if __name__ == "__main__":
    unittest.main()
