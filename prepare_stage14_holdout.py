"""Materialize the Stage-14 blind holdout only after its protocol is frozen."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prepare_stage12_holdout import load_exclusions, reservoir_select, sha256


ROOT = Path(__file__).resolve().parent


def validate(protocol: dict, protocol_path: Path) -> None:
    if protocol.get("status") != "frozen" or protocol.get("test_accessed_at_freeze") is not False:
        raise ValueError("A sealed frozen protocol is required")
    if protocol_path.parent != ROOT:
        raise ValueError("Protocol must be stored at project root")
    checkpoint = ROOT / protocol["checkpoint"]
    if not checkpoint.exists() or sha256(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint is missing or changed")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="STAGE14_TOKCOM_FINAL_PROTOCOL.json")
    parser.add_argument("--validation-parquet", required=True)
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    if not protocol_path.exists():
        parser.error("Frozen protocol does not exist; holdout remains sealed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate(protocol, protocol_path)
    source = Path(args.validation_parquet).resolve()
    if sha256(source) != protocol["source_validation_sha256"]:
        raise ValueError("Source validation parquet hash mismatch")
    out = ROOT / protocol["holdout_directory"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite materialized holdout: {out}")
    temporary_out = out.with_name(out.name + ".tmp")
    if temporary_out.exists():
        shutil.rmtree(temporary_out)

    import pyarrow.parquet as pq

    excluded_files, excluded_pixels, verified = load_exclusions(protocol["exclusion_manifests"])
    parquet = pq.ParquetFile(source)
    metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
    names = metadata["info"]["features"]["label"]["names"]
    if names != protocol["class_names"]:
        raise ValueError("Class vocabulary mismatch")

    def rows():
        for batch in parquet.iter_batches(batch_size=64):
            yield from batch.to_pylist()

    groups, eligible, excluded_count = reservoir_select(
        rows(), len(names), protocol["per_class"], protocol["selection_seed"], excluded_files, excluded_pixels
    )
    selected = [item for group in groups for item in group]
    if len({item["sha256"] for item in selected}) != len(selected):
        raise ValueError("Duplicate file content within selected blind holdout")
    if len({item["pixel_sha256"] for item in selected}) != len(selected):
        raise ValueError("Duplicate decoded pixels within selected blind holdout")
    temporary_out.mkdir(parents=True)
    records = []
    for label, group in enumerate(groups):
        for item in sorted(group, key=lambda row: row["sha256"]):
            destination = temporary_out / "images/test" / str(label) / f"{item['sha256'][:20]}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.pop("raw"))
            records.append({"image": destination.relative_to(temporary_out).as_posix(), "label": label, **item})
    test_manifest = temporary_out / "test.jsonl"
    test_manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "dataset": "Stage-14 preregistered disjoint Imagenette blind holdout",
        "class_names": names,
        "split_counts": {"test": len(records)},
        "protocol_sha256": sha256(protocol_path),
        "test_manifest_sha256": sha256(test_manifest),
        "cross_history_file_or_pixel_overlap": 0,
        "within_holdout_file_or_pixel_duplicates": 0,
        "test_materialized": True,
        "model_evaluation_accessed": False,
    }
    (temporary_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    identity = {
        **manifest,
        "source_validation_sha256": sha256(source),
        "selection_seed": protocol["selection_seed"],
        "per_class": protocol["per_class"],
        "eligible_per_class_after_exclusion": eligible,
        "excluded_source_rows": excluded_count,
        "verified_exclusion_manifests": verified,
        "holdout_materialized_after_freeze": True,
    }
    (temporary_out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    temporary_out.replace(out)
    print(json.dumps({"records": len(records), "test_manifest_sha256": identity["test_manifest_sha256"]}))


if __name__ == "__main__":
    main()
