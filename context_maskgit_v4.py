"""Stronger class-conditioned MaskGIT receiver for the TokCom case study.

V4 preserves every trained V3 weight and adds zero-initialized conditional
FiLM plus a class-specific code prior. The zero initialization makes the
warm-start logits exactly equal to V3 before further training.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F
from torch import nn

from context_maskgit_v3 import ContextMaskGITv3
from models import erase_packets


class ContextMaskGITv4(ContextMaskGITv3):
    def __init__(
        self,
        d: int = 128,
        layers: int = 4,
        local_blocks: int = 2,
        classes: int = 10,
        context_dropout: float = 0.15,
        context_prior_weight: float = 0.15,
    ) -> None:
        super().__init__(d, layers, local_blocks, classes, context_dropout)
        if context_prior_weight < 0:
            raise ValueError("context_prior_weight must be nonnegative")
        self.context_prior_weight = float(context_prior_weight)
        self.condition_film = nn.Linear(d, 2 * d)
        self.condition_prior = nn.Linear(d, self.codes, bias=False)
        nn.init.zeros_(self.condition_film.weight)
        nn.init.zeros_(self.condition_film.bias)
        nn.init.zeros_(self.condition_prior.weight)
        self.config.update(
            {
                "architecture": "additive_context_plus_output_film_and_code_prior",
                "context_prior_weight": self.context_prior_weight,
            }
        )

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.tokens:
            raise ValueError("Expected [batch,256] token IDs")
        labels = self._labels(labels, len(ids), ids.device)
        context_vector = self.condition(labels)
        context = context_vector[:, None, None, :]
        x = self.embedding(ids).view(-1, 16, 16, self.d) + self.row + self.column + context
        x = self.local(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(-1, self.tokens, self.d)
        x = self.norm(self.global_net(x))
        scale, shift = self.condition_film(context_vector).chunk(2, dim=-1)
        x = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.head(x) + self.condition_prior(context_vector)[:, None, :]

    def loss_components(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        packet_tokens: int = 4,
        per: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        clean_labels = self._labels(labels, len(ids), ids.device)
        used_labels = clean_labels.clone()
        if self.training and self.context_dropout:
            dropped = torch.rand(len(used_labels), device=used_labels.device) < self.context_dropout
            used_labels[dropped] = self.no_context_id
        missing = erase_packets(ids, per, packet_tokens)
        logits = self(ids.masked_fill(missing, self.codes), used_labels)
        masked_ce = F.cross_entropy(logits[missing], ids[missing])
        class_prior = self.condition_prior(self.condition(clean_labels))[:, None, :].expand(-1, self.tokens, -1)
        context_prior_ce = F.cross_entropy(class_prior[missing], ids[missing])
        total = masked_ce + self.context_prior_weight * context_prior_ce
        return total, missing, used_labels, {"masked_ce": masked_ce, "context_prior_ce": context_prior_ce}

    def loss(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        packet_tokens: int = 4,
        per: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total, missing, used_labels, _ = self.loss_components(ids, labels, packet_tokens, per)
        return total, missing, used_labels


def initialize_context_maskgit_v4_from_v3(
    checkpoint: Mapping,
    context_prior_weight: float = 0.15,
    device: str | torch.device = "cpu",
) -> ContextMaskGITv4:
    if checkpoint.get("model") != "context_maskgit_v3":
        raise ValueError("Expected ContextMaskGITv3 checkpoint")
    config = checkpoint["config"]
    model = ContextMaskGITv4(
        d=int(config["d"]),
        layers=int(config["layers"]),
        local_blocks=int(config["local_blocks"]),
        classes=int(config["classes"]),
        context_dropout=float(config["context_dropout"]),
        context_prior_weight=context_prior_weight,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    expected = {
        "condition_film.weight",
        "condition_film.bias",
        "condition_prior.weight",
    }
    if unexpected or set(missing) != expected:
        raise ValueError(f"Unexpected V3/V4 state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def load_context_maskgit_v4(checkpoint: Mapping, device: str | torch.device = "cpu") -> ContextMaskGITv4:
    if checkpoint.get("model") != "context_maskgit_v4":
        raise ValueError("Expected ContextMaskGITv4 checkpoint")
    config = checkpoint["config"]
    model = ContextMaskGITv4(
        d=int(config["d"]),
        layers=int(config["layers"]),
        local_blocks=int(config["local_blocks"]),
        classes=int(config["classes"]),
        context_dropout=float(config["context_dropout"]),
        context_prior_weight=float(config["context_prior_weight"]),
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    return model
