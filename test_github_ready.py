import unittest
from pathlib import Path

from audit_github_ready import FORBIDDEN_ROOTS, WEIGHT_ALLOWLIST


class GitHubReadinessTests(unittest.TestCase):
    def test_generated_roots_are_forbidden(self):
        self.assertTrue({"data", "runs", "pretrained", "work"}.issubset(FORBIDDEN_ROOTS))

    def test_weight_allowlist_is_narrow(self):
        self.assertTrue(WEIGHT_ALLOWLIST.fullmatch("third_party/perceptual_similarity/lpips/weights/v0.1/squeeze.pth"))
        self.assertFalse(WEIGHT_ALLOWLIST.fullmatch("runs/stage18/best.pth"))
        self.assertFalse(WEIGHT_ALLOWLIST.fullmatch("third_party/arbitrary/model.pth"))

    def test_gitignore_covers_runtime_artifacts(self):
        ignore = (Path(__file__).resolve().parent / ".gitignore").read_text(encoding="utf-8").splitlines()
        for entry in (
            "runs/",
            "data/",
            "pretrained/",
            "work/",
            "*.pt",
            "*.safetensors",
            "*.parquet",
            "STAGE*_REPORT.html",
        ):
            self.assertIn(entry, ignore)


if __name__ == "__main__":
    unittest.main()
