"""Fail-closed audit for the lightweight GitHub source deliverable."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "GITHUB_READINESS.json"
FORBIDDEN_ROOTS = {"data", "runs", "pretrained", "work", ".venv", "__pycache__", ".vscode"}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".parquet", ".part", ".zip", ".tar", ".gz", ".log"
}
WEIGHT_ALLOWLIST = re.compile(r"^third_party/perceptual_similarity/lpips/weights/v0\.[01]/(alex|squeeze|vgg)\.pth$")
SECRET_PATTERNS = {
    "private_key": re.compile(rb"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    "generic_secret_assignment": re.compile(
        rb"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*=\s*['\"][^'\"\r\n]{8,}"
    ),
}


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    names = [item.decode("utf-8") for item in git("ls-files", "-z").split(b"\0") if item]
    violations: list[dict] = []
    files: list[dict] = []
    secret_hits: list[dict] = []
    for name in names:
        relative = PurePosixPath(name)
        path = ROOT / name
        if not path.is_file():
            violations.append({"path": name, "reason": "tracked path is not a regular file"})
            continue
        size = path.stat().st_size
        files.append({"path": name, "bytes": size, "sha256": sha256(path)})
        if relative.parts and relative.parts[0].lower() in FORBIDDEN_ROOTS:
            violations.append({"path": name, "reason": "tracked generated/data root"})
        suffix = relative.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES and not WEIGHT_ALLOWLIST.fullmatch(name):
            violations.append({"path": name, "reason": f"tracked forbidden artifact suffix {suffix}"})
        if size > 10 * 1024 * 1024:
            violations.append({"path": name, "reason": "tracked file exceeds 10 MiB lightweight limit"})
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".pth"}:
            content = path.read_bytes()
            for pattern_name, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    secret_hits.append({"path": name, "pattern": pattern_name})
    if secret_hits:
        violations.extend({"path": row["path"], "reason": f"possible secret: {row['pattern']}"} for row in secret_hits)
    remote_lines = git("remote", "-v").decode("utf-8").splitlines()
    payload = {
        "status": "passed" if not violations else "failed",
        "tracked_files": len(files),
        "tracked_bytes": sum(row["bytes"] for row in files),
        "largest_tracked_file": max(files, key=lambda row: row["bytes"]) if files else None,
        "allowed_third_party_weight_files": sorted(
            row["path"] for row in files if WEIGHT_ALLOWLIST.fullmatch(row["path"])
        ),
        "forbidden_roots": sorted(FORBIDDEN_ROOTS),
        "forbidden_suffixes": sorted(FORBIDDEN_SUFFIXES),
        "secret_patterns_checked": sorted(SECRET_PATTERNS),
        "remote_lines": remote_lines,
        "violations": violations,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
