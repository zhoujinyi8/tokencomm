"""Text-conditioned MaskGIT receiver for 101-class scale transfer.

V5 leaves the frozen V4 implementation untouched. It replaces the learned
class lookup with fixed normalized CLIP text features plus a trainable linear
adapter. The adapter is zero-initialized around V4's no-context vector, so a
warm-started V5 exactly reproduces V4 unconditional logits before training.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from context_maskgit_v4 import ContextMaskGITv4, load_context_maskgit_v4


def class_names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


class TextConditionAdapter(nn.Module):
    def __init__(self, text_embeddings: torch.Tensor, output_dim: int, null: torch.Tensor | None = None) -> None:
        super().__init__()
        if text_embeddings.ndim != 2 or len(text_embeddings) < 2:
            raise ValueError("Expected [classes,text_dim] embeddings for at least two classes")
        if not torch.isfinite(text_embeddings).all():
            raise ValueError("Class text embeddings must be finite")
        normalized = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
        self.register_buffer("text_embeddings", normalized)
        self.projection = nn.Linear(normalized.shape[1], output_dim, bias=False)
        nn.init.zeros_(self.projection.weight)
        initial = torch.zeros(output_dim) if null is None else null.detach().float().clone()
        if initial.shape != (output_dim,):
            raise ValueError("Null context has the wrong dimensionality")
        self.null = nn.Parameter(initial)

    @property
    def classes(self) -> int:
        return len(self.text_embeddings)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        if bool(((labels < 0) | (labels > self.classes)).any()):
            raise ValueError("Text-context label is outside the configured vocabulary")
        output = self.null.expand(len(labels), -1).clone()
        conditioned = labels < self.classes
        if bool(conditioned.any()):
            output[conditioned] = output[conditioned] + self.projection(self.text_embeddings[labels[conditioned]])
        return output


class ContextMaskGITv5(ContextMaskGITv4):
    def __init__(
        self,
        class_text_embeddings: torch.Tensor,
        class_names: Sequence[str],
        d: int = 128,
        layers: int = 4,
        local_blocks: int = 2,
        context_dropout: float = 0.15,
        context_prior_weight: float = 0.15,
        null_context: torch.Tensor | None = None,
    ) -> None:
        names = list(class_names)
        if len(names) != len(class_text_embeddings) or len(set(names)) != len(names):
            raise ValueError("Class names must be unique and align with text embeddings")
        super().__init__(d, layers, local_blocks, len(names), context_dropout, context_prior_weight)
        self.condition = TextConditionAdapter(class_text_embeddings, d, null_context)
        self.class_names = names
        self.config.update({
            "architecture": "clip_text_context_plus_output_film_and_code_prior",
            "classes": len(names),
            "class_names": names,
            "class_names_sha256": class_names_sha256(names),
            "text_embedding_dim": int(class_text_embeddings.shape[1]),
            "condition": "reliable_class_name_clip_text_embedding",
        })


def initialize_context_maskgit_v5_from_v4(
    checkpoint: Mapping,
    class_text_embeddings: torch.Tensor,
    class_names: Sequence[str],
    device: str | torch.device = "cpu",
) -> ContextMaskGITv5:
    source = load_context_maskgit_v4(checkpoint, device="cpu").eval()
    config = checkpoint["config"]
    target = ContextMaskGITv5(
        class_text_embeddings=class_text_embeddings,
        class_names=class_names,
        d=int(config["d"]),
        layers=int(config["layers"]),
        local_blocks=int(config["local_blocks"]),
        context_dropout=float(config["context_dropout"]),
        context_prior_weight=float(config["context_prior_weight"]),
        null_context=source.condition.weight[source.no_context_id],
    )
    target_state = target.state_dict()
    copied = []
    for name, value in source.state_dict().items():
        if name == "condition.weight":
            continue
        if name not in target_state or target_state[name].shape != value.shape:
            raise ValueError(f"Unexpected V4/V5 state mismatch: {name}")
        target_state[name] = value
        copied.append(name)
    target.load_state_dict(target_state)
    target.config["warm_start_source_model"] = "context_maskgit_v4"
    target.config["warm_start_copied_tensors"] = len(copied)
    return target.to(device)


def load_context_maskgit_v5(checkpoint: Mapping, device: str | torch.device = "cpu") -> ContextMaskGITv5:
    if checkpoint.get("model") != "context_maskgit_v5":
        raise ValueError("Expected ContextMaskGITv5 checkpoint")
    config = checkpoint["config"]
    embeddings = checkpoint["state"]["condition.text_embeddings"]
    model = ContextMaskGITv5(
        class_text_embeddings=embeddings,
        class_names=config["class_names"],
        d=int(config["d"]),
        layers=int(config["layers"]),
        local_blocks=int(config["local_blocks"]),
        context_dropout=float(config["context_dropout"]),
        context_prior_weight=float(config["context_prior_weight"]),
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    if class_names_sha256(model.class_names) != config["class_names_sha256"]:
        raise ValueError("V5 class-name identity mismatch")
    return model
