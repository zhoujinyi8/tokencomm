"""Create and verify a resumable paused-project archive on the D: drive."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path("D:/")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE = OUTPUT_DIR / f"TokenCom_TaiChi_Reproduction_Paused_{STAMP}.zip"
PARTIAL = ARCHIVE.with_suffix(".zip.partial")
CHECKSUM = Path(f"{ARCHIVE}.sha256")
TOP = "tokencom_reproduction"

SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".partial", ".log"}


def should_include(relative: Path) -> bool:
    parts = relative.parts
    if any(part in SKIP_DIR_NAMES for part in parts):
        return False
    if relative.suffix.lower() in SKIP_SUFFIXES:
        return False
    # Public source images and per-image token arrays can be recreated from the
    # retained manifests, source parquet, model weights, and preprocessing code.
    if parts and parts[0] == "data":
        if "images" in parts:
            return False
        if "token_cache" in parts and relative.suffix.lower() == ".npy":
            return False
    # Retain only the source parquet needed to rematerialize the held-out data.
    if parts and parts[0] == "work":
        return relative.as_posix() == "work/stage12_source/validation-00000-of-00001.parquet"
    return True


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and should_include(path.relative_to(ROOT)):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = iter_files()
    total_bytes = sum(path.stat().st_size for path in files)
    manifest_path = ROOT / "PACKAGE_MANIFEST.json"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(ROOT),
        "archive_target": str(ARCHIVE),
        "pause_state": {
            "scheduled_tasks_disabled": [
                "TokenCom-Stage13-Orchestrator",
                "TokenCom-Stage13-4Hour-Progress",
            ],
            "related_training_workers_running": 0,
        },
        "research_state": {
            "stage13_training": "completed 6000/6000",
            "stage13_best_step": 6000,
            "stage13_best_checkpoint_sha256": "f8b8c0ab73177b048d6bb33522b257d07e8c6c48cecfc092d8a183e1152bd919",
            "stage13_physical_dev": "paused after 1/24 conditions",
            "stage14_protocol_frozen": False,
            "stage14_holdout_materialized": False,
            "stage14_test_accessed": False,
        },
        "included": [
            "source code and project metadata",
            "reports, protocols, audits, and figures",
            "runs/ checkpoints and lightweight results",
            "pretrained/ weights required for resumption",
            "third_party/ source dependencies",
            "data/ manifests and aggregate feature/token tensors",
            "work/stage12_source/validation-00000-of-00001.parquet",
        ],
        "excluded_recreatable_or_transient": [
            ".git metadata and Python/tool caches",
            "public raw images under data/**/images",
            "per-image data/**/token_cache/*.npy files",
            "transient logs, temporary files, and other work/ content",
        ],
        "resume_instructions": "PAUSE_AND_RESUME_STATUS.md",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Refresh the list because PACKAGE_MANIFEST.json has just been created.
    files = iter_files()
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"ARCHIVE={ARCHIVE}", flush=True)
    print(f"FILES={len(files)} UNCOMPRESSED_BYTES={total_bytes}", flush=True)

    if PARTIAL.exists():
        PARTIAL.unlink()
    started = time.monotonic()
    processed = 0
    with zipfile.ZipFile(
        PARTIAL,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{TOP}/{relative}")
            processed += path.stat().st_size
            if index % 200 == 0 or processed // (256 * 1024 * 1024) != (processed - path.stat().st_size) // (256 * 1024 * 1024):
                pct = 100.0 * processed / total_bytes if total_bytes else 100.0
                print(
                    f"PROGRESS files={index}/{len(files)} bytes={processed}/{total_bytes} pct={pct:.1f}",
                    flush=True,
                )

    os.replace(PARTIAL, ARCHIVE)
    print(f"COMPRESSED_BYTES={ARCHIVE.stat().st_size}", flush=True)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
        forbidden = [
            name
            for name in names
            if "/.git/" in name
            or "/__pycache__/" in name
            or ("/images/" in name and name.startswith(f"{TOP}/data/"))
            or ("/token_cache/" in name and name.lower().endswith(".npy"))
        ]
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC verification failed for {bad_member}")
    if forbidden:
        raise RuntimeError(f"Excluded files entered the archive: {forbidden[:5]}")

    digest = sha256(ARCHIVE)
    CHECKSUM.write_text(f"{digest}  {ARCHIVE.name}\n", encoding="ascii")
    elapsed = time.monotonic() - started
    print(f"ZIP_TEST=PASS ENTRIES={len(names)}", flush=True)
    print(f"SHA256={digest}", flush=True)
    print(f"CHECKSUM_FILE={CHECKSUM}", flush=True)
    print(f"ELAPSED_SECONDS={elapsed:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
