"""Materialize a new Imagenette holdout only after a frozen Stage-12 protocol.

This script is intentionally unusable before Stage-11 completes and a protocol
binds one checkpoint, its SHA-256, the upstream validation parquet, the exact
exclusion manifests, the sample size, and the deterministic selection seed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_digest(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()


def validate_frozen_protocol(protocol: dict, protocol_path: Path) -> None:
    required = {
        "status",
        "stage11_final_summary",
        "selected_variant",
        "checkpoint",
        "checkpoint_sha256",
        "source_validation_sha256",
        "selection_seed",
        "per_class",
        "class_names",
        "exclusion_manifests",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"Protocol is missing fields: {missing}")
    if protocol["status"] != "frozen":
        raise ValueError("Protocol status must be frozen before opening holdout data")
    final_summary = ROOT / protocol["stage11_final_summary"]
    if not final_summary.exists():
        raise FileNotFoundError("Stage-11 final summary does not exist")
    final = json.loads(final_summary.read_text(encoding="utf-8"))
    if final.get("all_completed") is not True:
        raise ValueError("Stage-11 is not complete")
    variants = {row["name"] for row in final["variants"]}
    if protocol["selected_variant"] not in variants:
        raise ValueError("Selected variant is not present in the completed Stage-11 summary")
    checkpoint = ROOT / protocol["checkpoint"]
    if not checkpoint.exists() or sha256(checkpoint).lower() != protocol["checkpoint_sha256"].lower():
        raise ValueError("Frozen checkpoint is missing or its SHA-256 does not match")
    if int(protocol["per_class"]) <= 0 or len(protocol["class_names"]) < 2:
        raise ValueError("Invalid class or sample count in protocol")
    if protocol_path.resolve().parent != ROOT.resolve():
        raise ValueError("Protocol must be stored at the project root for auditability")


def load_exclusions(entries: list[dict]) -> tuple[set[str], set[str], list[dict]]:
    file_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    verified = []
    for entry in entries:
        path = ROOT / entry["path"]
        actual = sha256(path)
        if actual.lower() != entry["sha256"].lower():
            raise ValueError(f"Exclusion manifest hash mismatch: {entry['path']}")
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            file_hashes.add(row["sha256"])
            pixel_hashes.add(row["pixel_sha256"])
            count += 1
        verified.append({"path": entry["path"], "sha256": actual, "records": count})
    return file_hashes, pixel_hashes, verified


def reservoir_select(rows, class_count: int, per_class: int, seed: int, excluded_files, excluded_pixels):
    rng = random.Random(seed)
    groups = [[] for _ in range(class_count)]
    eligible = [0] * class_count
    excluded = 0
    for row in rows:
        raw = row["image"]["bytes"]
        digest = hashlib.sha256(raw).hexdigest()
        pixels = pixel_digest(raw)
        if digest in excluded_files or pixels in excluded_pixels:
            excluded += 1
            continue
        label = int(row["label"])
        if not 0 <= label < class_count:
            raise ValueError(f"Unexpected label: {label}")
        eligible[label] += 1
        index = rng.randrange(eligible[label])
        item = {
            "raw": raw,
            "label": label,
            "sha256": digest,
            "pixel_sha256": pixels,
            "source_path": row["image"].get("path"),
        }
        if len(groups[label]) < per_class:
            groups[label].append(item)
        elif index < per_class:
            groups[label][index] = item
    if any(len(group) != per_class for group in groups):
        raise ValueError(f"Insufficient unseen examples: eligible per class={eligible}")
    return groups, eligible, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="STAGE12_FINAL_TEST_PROTOCOL.json")
    parser.add_argument("--validation-parquet", required=True)
    parser.add_argument("--out", default="data/stage12_final_holdout")
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    if not protocol_path.exists():
        parser.error("Frozen protocol does not exist; holdout remains sealed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol, protocol_path)
    source = Path(args.validation_parquet).resolve()
    if sha256(source).lower() != protocol["source_validation_sha256"].lower():
        raise ValueError("Upstream validation parquet SHA-256 mismatch")
    out = (ROOT / args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite materialized holdout: {out}")

    import pyarrow.parquet as pq

    excluded_files, excluded_pixels, verified = load_exclusions(protocol["exclusion_manifests"])
    parquet = pq.ParquetFile(source)
    metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
    upstream_names = metadata["info"]["features"]["label"]["names"]
    if upstream_names != protocol["class_names"]:
        raise ValueError("Upstream class vocabulary does not match the frozen protocol")

    def source_rows():
        for batch in parquet.iter_batches(batch_size=64):
            yield from batch.to_pylist()

    groups, eligible, excluded_count = reservoir_select(
        source_rows(), len(upstream_names), int(protocol["per_class"]), int(protocol["selection_seed"]),
        excluded_files, excluded_pixels,
    )
    out.mkdir(parents=True)
    records = []
    for label, group in enumerate(groups):
        for item in sorted(group, key=lambda row: row["sha256"]):
            dst = out / "images" / "test" / str(label) / f"{item['sha256'][:20]}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(item.pop("raw"))
            records.append(
                {
                    "image": dst.relative_to(out).as_posix(),
                    "label": label,
                    "question": "What is the main object in this image?",
                    "answer": upstream_names[label].lower(),
                    **item,
                }
            )
    manifest_path = out / "test.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    identity = {
        "purpose": "Stage-12 preregistered final holdout",
        "protocol": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "source_validation_sha256": sha256(source),
        "test_manifest_sha256": sha256(manifest_path),
        "selection_seed": int(protocol["selection_seed"]),
        "per_class": int(protocol["per_class"]),
        "records": len(records),
        "eligible_per_class_after_exclusion": eligible,
        "excluded_source_rows": excluded_count,
        "verified_exclusion_manifests": verified,
        "cross_history_file_or_pixel_overlap": 0,
        "holdout_materialized_after_freeze": True,
        "model_evaluation_accessed": False,
    }
    (out / "identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(identity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
