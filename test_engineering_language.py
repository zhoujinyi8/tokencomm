import tempfile
import unittest
from pathlib import Path

from audit_engineering_language import find_violations


class EngineeringLanguageAuditTests(unittest.TestCase):
    def test_repository_markdown_passes(self):
        root = Path(__file__).parent
        self.assertEqual(find_violations(root), [])

    def test_reports_file_line_and_phrase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "REPORT.md").write_text(
                "# 报告\n值得注意的是，结果很好。\n", encoding="utf-8"
            )
            violations = find_violations(root)

        self.assertEqual(len(violations), 1)
        path, line_number, phrase, _ = violations[0]
        self.assertEqual(path, Path("REPORT.md"))
        self.assertEqual(line_number, 2)
        self.assertEqual(phrase, "值得注意的是")


if __name__ == "__main__":
    unittest.main()
