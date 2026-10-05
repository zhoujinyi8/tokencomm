"""Cache Stage-12 dual-vision features only after the final protocol is frozen."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights

from cache_dual_vision import extract, load_backbones
from prepare_stage12_holdout import ROOT, pixel_digest, sha256, validate_frozen_protocol


@torch.inference_mode()
def main() -> None:
    protocol_path = ROOT / "STAGE12_FINAL_TEST_PROTOCOL.json"
    if not protocol_path.exists():
        raise FileNotFoundError("Frozen Stage-12 protocol does not exist; test remains sealed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol, protocol_path)
    holdout = ROOT / protocol["holdout_directory"]
    manifest_path = holdout / "test.jsonl"
    identity_path = holdout / "identity.json"
    if not manifest_path.exists() or not identity_path.exists():
        raise FileNotFoundError("Materialize the frozen holdout before caching features")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Holdout/protocol hash mismatch")
    if identity["test_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Holdout manifest hash mismatch")
    if identity["model_evaluation_accessed"] is not False:
        raise ValueError("Unexpected pre-existing evaluation access marker")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    expected = int(protocol["per_class"]) * len(protocol["class_names"])
    if len(records) != expected or identity["records"] != expected:
        raise ValueError("Frozen holdout count mismatch")
    counts = torch.bincount(torch.tensor([row["label"] for row in records]), minlength=len(protocol["class_names"]))
    if not torch.equal(counts, torch.full_like(counts, int(protocol["per_class"]))):
        raise ValueError(f"Holdout is not class balanced: {counts.tolist()}")
    # Recheck both encoded bytes and decoded pixels before feature extraction.
    for row in records:
        path = holdout / row["image"]
        raw = path.read_bytes()
        if sha256(path) != row["sha256"] or pixel_digest(raw) != row["pixel_sha256"]:
            raise ValueError(f"Holdout image identity mismatch: {row['image']}")

    out = holdout / "dual_vision_features"
    out.mkdir(exist_ok=True)
    destination = out / "test.pt"
    if destination.exists():
        raise FileExistsError("Stage-12 feature cache already exists; refusing overwrite")
    torch.set_num_threads(4)
    vit, conv, vit_hash, conv_hash = load_backbones()
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    vit_features, conv_features = [], []
    batch_size = int(protocol["feature_cache_batch_size"])
    for start in range(0, len(records), batch_size):
        images = []
        for row in records[start : start + batch_size]:
            with Image.open(holdout / row["image"]) as image:
                images.append(transform(image.convert("RGB")))
        v, c = extract(vit, conv, torch.stack(images))
        vit_features.append(v.half())
        conv_features.append(c.half())
        if start % 40 == 0:
            print(start, "/", len(records), flush=True)
    cache_identity = {
        "purpose": "Stage-12 preregistered final holdout features",
        "protocol_sha256": sha256(protocol_path),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "test_manifest_sha256": sha256(manifest_path),
        "vit_weight_sha256": vit_hash,
        "conv_weight_sha256": conv_hash,
        "preprocessing": str(transform),
        "test_opened": True,
        "model_evaluation_accessed": False,
        "used_for_training": False,
    }
    torch.save(
        {
            "vit": torch.cat(vit_features),
            "conv": torch.cat(conv_features),
            "labels": torch.tensor([row["label"] for row in records]),
            "records": records,
            "identity": cache_identity,
        },
        destination,
    )
    (out / "identity.json").write_text(json.dumps(cache_identity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cache_identity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
