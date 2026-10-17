import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class GoalFiveHourTests(unittest.TestCase):
    def test_status_generator_and_hidden_dispatch(self):
        subprocess.run([sys.executable, str(ROOT / "goal_5hour_status.py")], check=True, capture_output=True, text=True)
        status = json.loads((ROOT / "GOAL_5H_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(status["cadence_hours"], 5)
        self.assertEqual(status["active_minutes"], 20)
        self.assertEqual(status["stage19_audit"], "passed")
        self.assertTrue(status["stage20_prepared"])
        self.assertFalse(status["stage20_test_accessed"])
        script = (ROOT / "advance_goal_5hour.ps1").read_text(encoding="utf-8")
        wrapper = (ROOT / "advance_goal_5hour_hidden.vbs").read_text(encoding="utf-8")
        installer = (ROOT / "install_goal_5hour_task.ps1").read_text(encoding="utf-8")
        self.assertIn("goal_5hour_status.py", script)
        self.assertIn("cache_stage21_food101_tokens.py", script)
        self.assertIn("$taskBudgetSeconds = 1200", script)
        self.assertIn("--max-seconds 1050", script)
        self.assertIn("finalize_stage21_token_cache.py", script)
        self.assertIn("freeze_stage22_protocol.py", script)
        self.assertIn("train_stage22_contextual_tokcom.py", script)
        self.assertIn("finalize_stage22.py", script)
        self.assertIn("freeze_stage23_link_protocol.py", script)
        self.assertIn("evaluate_stage23_v5_link.py", script)
        self.assertIn("finalize_stage23.py", script)
        self.assertIn("-WindowStyle Hidden", wrapper)
        self.assertIn("shell.Run(command, 0, True)", wrapper)
        self.assertIn("WScript.Quit exitCode", wrapper)
        self.assertIn("-RepetitionInterval (New-TimeSpan -Hours 5)", installer)
        self.assertIn("-ExecutionTimeLimit (New-TimeSpan -Minutes 25)", installer)
        self.assertIn("-MultipleInstances IgnoreNew", installer)
        self.assertIn("-Priority 5", installer)


if __name__ == "__main__":
    unittest.main()
