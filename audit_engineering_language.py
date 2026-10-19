"""Check repository documentation for non-technical filler language."""

from __future__ import annotations

import argparse
from pathlib import Path


BANNED_PHRASES = (
    "值得注意的是",
    "需要强调的是",
    "令人印象深刻",
    "综上所述",
    "总而言之",
    "显而易见",
    "可以看出",
    "全面提升",
    "强大能力",
    "无缝衔接",
    "无缝恢复",
    "赋能",
)

EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules"}


def document_files(root: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("*.md", "*.html")
        for path in root.rglob(pattern)
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts)
    )


def find_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for path in document_files(root):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for phrase in BANNED_PHRASES:
                if phrase in line:
                    violations.append(
                        (path.relative_to(root), line_number, phrase, line.strip())
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查Markdown中的空泛、口号化或无法量化的固定表达。"
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.resolve()
    violations = find_violations(root)
    if violations:
        for path, line_number, phrase, line in violations:
            print(f"{path}:{line_number}: 禁用短语 {phrase!r}: {line}")
        return 1
    print(f"检查通过：{len(document_files(root))}个Markdown/HTML文档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
