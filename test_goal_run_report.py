import tempfile
import unittest
from pathlib import Path

from goal_run_report import snapshot


class GoalRunReportTests(unittest.TestCase):
    def test_snapshot_reports_progress_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "data/stage20_food101/token_cache_stage21"
            cache.mkdir(parents=True)
            (cache / "progress.json").write_text('{"cached_images": 64, "planned_images": 100, "complete": false, "test_accessed": false}')
            data = root / "data/stage20_food101"
            (data / "manifest.json").write_text('{"test_pixels_extracted": false, "test_accessed": false}')
            run = root / "runs/stage22_contextual_tokcom"
            run.mkdir(parents=True)
            (run / "history.jsonl").write_text('{"step": 1}\n{"step": 2}\n')
            result = snapshot(root)
            self.assertEqual(result["stage21_cached_images"], 64)
            self.assertEqual(result["stage22_step"], 2)
            self.assertFalse(result["test_accessed"])
            self.assertFalse(result["test_pixels_extracted"])


if __name__ == "__main__":
    unittest.main()
