import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class GoalFourHourTests(unittest.TestCase):
    def test_status_generator_and_hidden_dispatch(self):
        subprocess.run([sys.executable, str(ROOT / "goal_4hour_status.py")], check=True, capture_output=True, text=True)
        status = json.loads((ROOT / "GOAL_4H_STATUS.json").read_text(encoding="utf-8"))
        self.assertEqual(status["cadence_hours"], 4)
        self.assertEqual(status["stage19_audit"], "passed")
        self.assertTrue(status["stage20_prepared"])
        self.assertFalse(status["stage20_test_accessed"])
        script = (ROOT / "advance_goal_4hour.ps1").read_text(encoding="utf-8")
        wrapper = (ROOT / "advance_goal_4hour_hidden.vbs").read_text(encoding="utf-8")
        installer = (ROOT / "install_goal_4hour_task.ps1").read_text(encoding="utf-8")
        self.assertIn("goal_4hour_status.py", script)
        self.assertIn("cache_stage21_food101_tokens.py", script)
        self.assertIn("$taskBudgetSeconds = 12600", script)
        self.assertIn("--max-seconds 12000", script)
        self.assertIn("finalize_stage21_token_cache.py", script)
        self.assertIn("freeze_stage22_protocol.py", script)
        self.assertIn("train_stage22_contextual_tokcom.py", script)
        self.assertIn("-WindowStyle Hidden", wrapper)
        self.assertIn("shell.Run(command, 0, True)", wrapper)
        self.assertIn("WScript.Quit exitCode", wrapper)
        self.assertIn("-RepetitionInterval (New-TimeSpan -Hours 4)", installer)
        self.assertIn("-ExecutionTimeLimit (New-TimeSpan -Hours 3 -Minutes 45)", installer)
        self.assertIn("-MultipleInstances IgnoreNew", installer)


if __name__ == "__main__":
    unittest.main()
