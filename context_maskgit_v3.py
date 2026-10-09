"""Context-conditioned MaskGIT receiver for cross-modal TokCom experiments.

This module intentionally lives beside ``context_maskgit_v2`` instead of
changing it: Stage-14 binds the v2 implementation hash.  V3 adds reliable
class side information, classifier-free context dropout, and conditional
guidance while preserving the same 256-token/1024-code architecture.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from context_maskgit_v2 import LocalBlock
from models import erase_packets, transformer


class ContextMaskGITv3(nn.Module):
    codes = 1024
    tokens = 256

    def __init__(
        self,
        d: int = 192,
        layers: int = 6,
        local_blocks: int = 3,
        classes: int = 10,
        context_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if d % 8:
            raise ValueError("d must divide eight groups")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if not 0.0 <= context_dropout <= 1.0:
            raise ValueError("context_dropout must be in [0, 1]")
        self.d = d
        self.classes = classes
        self.no_context_id = classes
        self.context_dropout = float(context_dropout)
        self.config = {
            "d": d,
            "layers": layers,
            "local_blocks": local_blocks,
            "classes": classes,
            "condition": "reliable_class_label",
            "context_dropout": self.context_dropout,
        }
        self.embedding = nn.Embedding(self.codes + 1, d)
        self.condition = nn.Embedding(classes + 1, d)
        self.row = nn.Parameter(torch.randn(1, 16, 1, d) * 0.02)
        self.column = nn.Parameter(torch.randn(1, 1, 16, d) * 0.02)
        self.local = nn.Sequential(*[LocalBlock(d) for _ in range(local_blocks)])
        self.global_net = transformer(d, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.codes)

    def _labels(self, labels: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if labels is None:
            return torch.full((batch,), self.no_context_id, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        if labels.shape != (batch,):
            raise ValueError("Expected one context label per image")
        if bool(((labels < 0) | (labels > self.no_context_id)).any()):
            raise ValueError("Context label is outside the configured vocabulary")
        return labels

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.tokens:
            raise ValueError("Expected [batch,256] token IDs")
        labels = self._labels(labels, len(ids), ids.device)
        context = self.condition(labels)[:, None, None, :]
        x = self.embedding(ids).view(-1, 16, 16, self.d) + self.row + self.column + context
        x = self.local(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(-1, self.tokens, self.d)
        return self.head(self.norm(self.global_net(x)))

    def guided_logits(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor | None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Classifier-free context guidance in logit space.

        A scale of zero is the unconditional receiver and one is the ordinary
        conditional receiver.  Values above one strengthen the class context.
        """
        if not math.isfinite(float(guidance_scale)) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if labels is None:
            return self(ids, None)
        conditional = self(ids, labels)
        if guidance_scale == 1.0:
            return conditional
        unconditional = self(ids, None)
        return unconditional + float(guidance_scale) * (conditional - unconditional)

    def token_likelihood(
        self,
        ids: torch.Tensor,
        missing: torch.Tensor,
        labels: torch.Tensor | None,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return maximum token likelihood and prediction at every position."""
        if missing.shape != ids.shape or missing.dtype != torch.bool:
            raise ValueError("missing must be a boolean mask with the token shape")
        masked = ids.masked_fill(missing, self.codes)
        return self.guided_logits(masked, labels, guidance_scale).softmax(-1).max(-1)

    @torch.no_grad()
    def recover_with_confidence(
        self,
        ids: torch.Tensor,
        missing: torch.Tensor,
        labels: torch.Tensor | None,
        steps: int = 8,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iteratively recover missing tokens and retain assignment likelihoods."""
        if steps < 1:
            raise ValueError("steps must be positive")
        if missing.shape != ids.shape or missing.dtype != torch.bool:
            raise ValueError("missing must be a boolean mask with the token shape")
        x = ids.masked_fill(missing, self.codes).clone()
        remaining = missing.clone()
        initial = missing.sum(1)
        assigned_confidence = torch.ones_like(ids, dtype=torch.float32)
        for step in range(steps):
            confidence, prediction = self.guided_logits(x, labels, guidance_scale).softmax(-1).max(-1)
            for batch_index in range(len(x)):
                indices = remaining[batch_index].nonzero().flatten()
                target = int(initial[batch_index] * math.cos((step + 1) / steps * math.pi / 2))
                take = len(indices) - target
                if take > 0:
                    chosen = indices[confidence[batch_index, indices].topk(take).indices]
                    x[batch_index, chosen] = prediction[batch_index, chosen]
                    assigned_confidence[batch_index, chosen] = confidence[batch_index, chosen]
                    remaining[batch_index, chosen] = False
        return x, assigned_confidence

    @torch.no_grad()
    def recover(
        self,
        ids: torch.Tensor,
        missing: torch.Tensor,
        labels: torch.Tensor | None,
        steps: int = 8,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.recover_with_confidence(ids, missing, labels, steps, guidance_scale)[0]

    def loss(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        packet_tokens: int = 4,
        per: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked-token loss with classifier-free context dropout."""
        labels = self._labels(labels, len(ids), ids.device).clone()
        if self.training and self.context_dropout:
            dropped = torch.rand(len(labels), device=labels.device) < self.context_dropout
            labels[dropped] = self.no_context_id
        missing = erase_packets(ids, per, packet_tokens)
        logits = self(ids.masked_fill(missing, self.codes), labels)
        return F.cross_entropy(logits[missing], ids[missing]), missing, labels


def initialize_context_maskgit_v3_from_v2(
    checkpoint: Mapping,
    classes: int = 10,
    context_dropout: float = 0.15,
    device: str | torch.device = "cpu",
) -> ContextMaskGITv3:
    """Warm-start V3 from a frozen V2 checkpoint without changing V2 files."""
    if checkpoint.get("model") != "context_maskgit_v2":
        raise ValueError("Expected ContextMaskGITv2 checkpoint")
    source_config = checkpoint["config"]
    model = ContextMaskGITv3(
        d=int(source_config["d"]),
        layers=int(source_config["layers"]),
        local_blocks=int(source_config["local_blocks"]),
        classes=classes,
        context_dropout=context_dropout,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    if unexpected or missing != ["condition.weight"]:
        raise ValueError(f"Unexpected V2/V3 state mismatch: missing={missing}, unexpected={unexpected}")
    # Zero is the exact no-context warm start; class rows get small distinct
    # perturbations so gradients can specialize immediately.
    with torch.no_grad():
        model.condition.weight.normal_(mean=0.0, std=0.02)
        model.condition.weight[model.no_context_id].zero_()
    return model


def load_context_maskgit_v3(checkpoint: Mapping, device: str | torch.device = "cpu") -> ContextMaskGITv3:
    if checkpoint.get("model") != "context_maskgit_v3":
        raise ValueError("Expected ContextMaskGITv3 checkpoint")
    config = checkpoint["config"]
    model = ContextMaskGITv3(
        d=int(config["d"]),
        layers=int(config["layers"]),
        local_blocks=int(config["local_blocks"]),
        classes=int(config["classes"]),
        context_dropout=float(config["context_dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    return model
