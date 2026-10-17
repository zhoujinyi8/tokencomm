"""Build pinned CLIP text features for the 101 Food-101 class names.

The output tensor follows ``manifest.json`` label order exactly.  Only the
text tower is instantiated from the already-audited full CLIP checkpoint; the
tokenizer is pinned to the checkpoint's Hugging Face repository revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/stage20_food101"
DATA_MANIFEST = DATA / "manifest.json"
CHECKPOINT = ROOT / "pretrained/clip-vit-base-patch16-pytorch_model.bin"
OUTPUT = DATA / "clip_class_embeddings.pt"
OUTPUT_MANIFEST = ROOT / "STAGE21_CLIP_CONTEXT_MANIFEST.json"
TOKENIZER_REPOSITORY = "openai/clip-vit-base-patch16"
TOKENIZER_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
PROMPT_TEMPLATE = "a photo of {}, a type of food"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def class_prompt(class_name: str) -> str:
    return PROMPT_TEMPLATE.format(class_name.replace("_", " "))


def build_text_model(checkpoint: Path):
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection

    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        projection_dim=512,
    )
    model = CLIPTextModelWithProjection(config)
    full_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    text_state = {
        name: value
        for name, value in full_state.items()
        if (name.startswith("text_model.") or name.startswith("text_projection."))
        and not name.endswith("position_ids")
    }
    model.load_state_dict(text_state, strict=True)
    return model.requires_grad_(False).eval()


def tokenizer_file_hashes(tokenizer) -> dict[str, str]:
    paths: set[Path] = set()
    for value in tokenizer.init_kwargs.values():
        if isinstance(value, str):
            path = Path(value)
            if path.is_file():
                paths.add(path)
    backend = getattr(tokenizer, "_tokenizer", None)
    # save_pretrained below is the canonical, version-stable audit surface.
    del backend
    audit_dir = ROOT / "pretrained/clip_tokenizer_stage21"
    audit_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(audit_dir)
    paths.update(path for path in audit_dir.iterdir() if path.is_file())
    return {path.name: sha256_file(path) for path in sorted(paths, key=lambda item: item.name)}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@torch.inference_mode()
def build(device: str = "cpu") -> dict:
    from transformers import CLIPTokenizerFast

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing pinned CLIP checkpoint: {CHECKPOINT}")
    source = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    classes = source["classes"]
    if source.get("status") != "prepared" or len(classes) != 101 or len(set(classes)) != 101:
        raise ValueError("Stage-20 manifest must contain 101 unique prepared classes")
    if source.get("test_accessed") or source.get("test_pixels_extracted"):
        raise ValueError("Stage-20 official test split is no longer sealed")

    cache = ROOT / "pretrained/hf_cache"
    tokenizer = CLIPTokenizerFast.from_pretrained(
        TOKENIZER_REPOSITORY,
        revision=TOKENIZER_REVISION,
        cache_dir=cache,
    )
    prompts = [class_prompt(name) for name in classes]
    tokens = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    model = build_text_model(CHECKPOINT).to(device)
    output = model(
        input_ids=tokens["input_ids"].to(device),
        attention_mask=tokens["attention_mask"].to(device),
    ).text_embeds
    embeddings = F.normalize(output.float(), dim=-1).cpu().contiguous()
    if embeddings.shape != (101, 512) or not torch.isfinite(embeddings).all():
        raise ValueError(f"Unexpected CLIP text tensor: {tuple(embeddings.shape)}")
    if not torch.allclose(embeddings.norm(dim=-1), torch.ones(101), atol=1e-6, rtol=0):
        raise ValueError("CLIP class embeddings are not unit normalized")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".pt.tmp")
    torch.save({"embeddings": embeddings, "class_names": classes, "prompts": prompts}, temporary)
    os.replace(temporary, OUTPUT)
    tokenizer_hashes = tokenizer_file_hashes(tokenizer)
    payload = {
        "stage": 21,
        "status": "prepared",
        "purpose": "fixed CLIP text conditioning for ContextMaskGIT V5",
        "source_data_manifest": "data/stage20_food101/manifest.json",
        "classes": len(classes),
        "class_names_sha256": sha256_json(classes),
        "prompt_template": PROMPT_TEMPLATE,
        "prompts_sha256": sha256_json(prompts),
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files_sha256": tokenizer_hashes,
        "checkpoint": "pretrained/clip-vit-base-patch16-pytorch_model.bin",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "embedding_file": "data/stage20_food101/clip_class_embeddings.pt",
        "embedding_file_sha256": sha256_file(OUTPUT),
        "embedding_tensor_sha256": sha256_tensor(embeddings),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "unit_norm_max_error": float((embeddings.norm(dim=-1) - 1).abs().max()),
        "test_pixels_extracted": False,
        "test_accessed": False,
    }
    atomic_json(OUTPUT_MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(build(args.device)))


if __name__ == "__main__":
    main()
