"""Download Food-101 and materialize only its official training images.

The official test split stays compressed and is represented only by filenames
until a later final-test protocol explicitly unlocks pixel extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "STAGE20_FOOD101_SCALE_PLAN.json"
DATA = ROOT / "data/stage20_food101"
ARCHIVE = DATA / "food-101.tar.gz"
PART = DATA / "food-101.tar.gz.part"
EXTRACTED = DATA / "extracted"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download(plan: dict) -> None:
    expected = int(plan["source"]["content_length"])
    DATA.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != expected:
            raise ValueError("Existing Food-101 archive has the wrong size")
        return
    offset = PART.stat().st_size if PART.exists() else 0
    if offset > expected:
        raise ValueError("Partial Food-101 download exceeds the frozen size")
    headers = {"User-Agent": "TokenCom-reproduction/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(plan["source"]["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        with PART.open(mode) as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
    if PART.stat().st_size != expected:
        raise RuntimeError(f"Food-101 download incomplete: {PART.stat().st_size}/{expected}")
    os.replace(PART, ARCHIVE)
    atomic_json(DATA / "archive_identity.json", {
        "bytes": expected,
        "sha256": sha256(ARCHIVE),
        "etag": plan["source"]["etag"],
        "last_modified": plan["source"]["last_modified"],
    })


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "food-101":
        raise ValueError(f"Unsafe Food-101 archive member: {name}")
    return path


def read_metadata() -> tuple[list[str], list[str]]:
    wanted = {"food-101/meta/train.txt", "food-101/meta/test.txt", "food-101/meta/classes.txt"}
    values: dict[str, str] = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            safe_name(member.name)
            if member.name in wanted:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Missing metadata payload: {member.name}")
                values[member.name] = stream.read().decode("utf-8")
    if set(values) != wanted:
        raise ValueError("Food-101 metadata members are incomplete")
    train = [line.strip() for line in values["food-101/meta/train.txt"].splitlines() if line.strip()]
    test = [line.strip() for line in values["food-101/meta/test.txt"].splitlines() if line.strip()]
    classes = [line.strip() for line in values["food-101/meta/classes.txt"].splitlines() if line.strip()]
    return train, test, classes


def split_train(train: list[str], classes: list[str], dev_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    train_out, dev_out = [], []
    for class_name in classes:
        rows = [name for name in train if name.split("/", 1)[0] == class_name]
        ranked = sorted(rows, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest())
        if len(ranked) != 750:
            raise ValueError(f"Expected 750 official training images for {class_name}, found {len(ranked)}")
        dev_out.extend(ranked[:dev_per_class])
        train_out.extend(ranked[dev_per_class:])
    return sorted(train_out), sorted(dev_out)


def extract_training_images(required: set[str]) -> None:
    remaining = set(required)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            path = safe_name(member.name)
            if member.name not in remaining:
                continue
            if not member.isfile():
                raise ValueError(f"Expected regular image file: {member.name}")
            target = EXTRACTED.joinpath(*path.parts)
            if target.exists() and target.stat().st_size == member.size:
                remaining.remove(member.name)
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read image member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            os.replace(temporary, target)
            remaining.remove(member.name)
    if remaining:
        raise ValueError(f"Food-101 archive omitted {len(remaining)} official training images")


def records(names: list[str], class_to_label: dict[str, int]) -> list[dict]:
    return [
        {
            "image": f"extracted/food-101/images/{name}.jpg",
            "label": class_to_label[name.split("/", 1)[0]],
            "class_name": name.split("/", 1)[0],
        }
        for name in names
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def prepare(plan: dict) -> None:
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != int(plan["source"]["content_length"]):
        raise FileNotFoundError("Run the Stage-20 Food-101 download first")
    train, test, classes = read_metadata()
    if len(classes) != 101 or len(train) != 75_750 or len(test) != 25_250:
        raise ValueError("Official Food-101 split counts changed")
    train_names, dev_names = split_train(train, classes, int(plan["split"]["dev_per_class"]), int(plan["split"]["seed"]))
    required = {f"food-101/images/{name}.jpg" for name in train}
    extract_training_images(required)
    class_to_label = {name: index for index, name in enumerate(classes)}
    write_jsonl(DATA / "train.jsonl", records(train_names, class_to_label))
    write_jsonl(DATA / "dev.jsonl", records(dev_names, class_to_label))
    atomic_json(DATA / "sealed_test_manifest.json", {
        "status": "sealed",
        "pixels_extracted": False,
        "used_for_training": False,
        "classes": len(classes),
        "images": len(test),
        "records": [{"name": name, "label": class_to_label[name.split("/", 1)[0]]} for name in test],
    })
    atomic_json(DATA / "manifest.json", {
        "status": "prepared",
        "approximation": "Food-101 101-class scale transfer; not ImageNet100",
        "archive_sha256": sha256(ARCHIVE),
        "classes": classes,
        "label_bits": 7,
        "train_images": len(train_names),
        "dev_images": len(dev_names),
        "sealed_test_images": len(test),
        "test_pixels_extracted": False,
        "test_accessed": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare", "all"), default="all", nargs="?")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen":
        raise ValueError("Stage-20 data plan must be frozen")
    if args.action in ("download", "all"):
        download(plan)
    if args.action in ("prepare", "all"):
        prepare(plan)


if __name__ == "__main__":
    main()
