"""ARQ and token-communication-efficiency utilities for TokCom.

The analytical metric follows the case-study definition in the TokCom paper:
T = 1 / (1 - PER), TCE = H*W / (T*N*log2(Q)).  The simulator additionally
supports confidence-gated selective retransmission using a contextual receiver.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from digital_link import bits_int, int_bits, transmit_packet


def analytical_arq_metrics(
    per: float,
    image_height: int = 256,
    image_width: int = 256,
    token_count: int = 256,
    codebook_size: int = 1024,
) -> dict[str, float]:
    if not 0.0 <= per < 1.0:
        raise ValueError("PER must be in [0, 1)")
    if min(image_height, image_width, token_count) < 1:
        raise ValueError("Image and token dimensions must be positive")
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two")
    token_bits = int(math.log2(codebook_size))
    transmissions = 1.0 / (1.0 - per)
    source_bits = token_count * token_bits
    return {
        "per": float(per),
        "expected_transmissions_per_packet": transmissions,
        "source_bits_per_image": float(source_bits),
        "source_bpp": source_bits / (image_height * image_width),
        "effective_bits_per_image": transmissions * source_bits,
        "effective_bpp": transmissions * source_bits / (image_height * image_width),
        "tce_pixels_per_token_bit": image_height * image_width / (transmissions * source_bits),
    }


@torch.inference_mode()
def transmit_tokens_with_arq(
    ids: np.ndarray,
    *,
    receiver: Any | None = None,
    labels: torch.Tensor | None = None,
    codebook_size: int = 1024,
    snr_db: float = 8.0,
    channel: str = "awgn",
    seed: int = 42,
    packet_tokens: int = 4,
    max_rounds: int = 4,
    policy: str = "all_failed",
    confidence_threshold: float = 0.5,
    recovery_steps: int = 8,
    guidance_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transmit tokens with conventional or confidence-selective ARQ.

    ``max_rounds`` includes the first transmission.  ``all_failed`` retries
    every CRC-failed packet.  ``confidence`` retries only failed packets whose
    mean contextual recovery likelihood is below ``confidence_threshold``.
    Packets trusted to semantic recovery remain erased until final inference.
    """
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected nonempty integer [B,N]")
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two")
    if packet_tokens < 1 or max_rounds < 1:
        raise ValueError("packet_tokens and max_rounds must be positive")
    if policy not in {"all_failed", "confidence"}:
        raise ValueError("policy must be all_failed or confidence")
    if policy == "confidence" and receiver is None:
        raise ValueError("confidence policy requires a contextual receiver")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if np.any(ids < 0) or np.any(ids >= codebook_size):
        raise ValueError("Token ID outside codebook")

    batch, tokens = ids.shape
    width = int(math.log2(codebook_size))
    rng = np.random.default_rng(seed)
    order = rng.permutation(tokens)
    groups = [order[start : start + packet_tokens] for start in range(0, tokens, packet_tokens)]
    received = np.zeros_like(ids)
    missing = np.ones_like(ids, dtype=bool)
    pending = {(image, packet) for image in range(batch) for packet in range(len(groups))}
    initial_packets = len(pending)
    attempts = 0
    successes = 0
    undetected_packets = 0
    round_rows: list[dict[str, Any]] = []

    def contextual_confidence() -> np.ndarray:
        token_tensor = torch.from_numpy(received).long()
        missing_tensor = torch.from_numpy(missing)
        used_labels = labels
        if used_labels is not None:
            used_labels = used_labels.to(dtype=torch.long)
        confidence, _ = receiver.token_likelihood(
            token_tensor, missing_tensor, used_labels, guidance_scale=guidance_scale
        )
        return confidence.cpu().numpy()

    for round_index in range(max_rounds):
        if not pending:
            break
        attempted_now = sorted(pending)
        pending_after: set[tuple[int, int]] = set()
        for image_index, packet_index in attempted_now:
            positions = groups[packet_index]
            payload = int_bits(ids[image_index, positions], width).reshape(-1)
            decoded, ok, detail = transmit_packet(payload, snr_db, rng, channel)
            attempts += 1
            if ok:
                decoded_ids = bits_int(decoded.reshape(-1, width))
                received[image_index, positions] = decoded_ids
                missing[image_index, positions] = False
                successes += 1
                undetected_packets += int(detail["payload_bit_errors"] > 0)
            else:
                pending_after.add((image_index, packet_index))

        requested = pending_after
        trusted_semantically = 0
        if policy == "confidence" and pending_after and round_index + 1 < max_rounds:
            confidence = contextual_confidence()
            requested = set()
            for image_index, packet_index in pending_after:
                score = float(confidence[image_index, groups[packet_index]].mean())
                if score < confidence_threshold:
                    requested.add((image_index, packet_index))
                else:
                    trusted_semantically += 1
        round_rows.append(
            {
                "round": round_index + 1,
                "attempted_packets": len(attempted_now),
                "crc_successes": len(attempted_now) - len(pending_after),
                "crc_failures": len(pending_after),
                "requested_retransmissions": len(requested) if round_index + 1 < max_rounds else 0,
                "trusted_to_semantic_recovery": trusted_semantically,
            }
        )
        pending = requested if round_index + 1 < max_rounds else pending_after

    restored = torch.from_numpy(received).long()
    assigned_confidence = torch.ones_like(restored, dtype=torch.float32)
    if receiver is not None and bool(missing.any()):
        restored, assigned_confidence = receiver.recover_with_confidence(
            restored,
            torch.from_numpy(missing),
            labels,
            steps=recovery_steps,
            guidance_scale=guidance_scale,
        )
    restored_np = restored.cpu().numpy()
    transmission_multiplier = attempts / initial_packets
    source_bits = tokens * width
    metrics = {
        "policy": policy,
        "rounds_used": len(round_rows),
        "max_rounds": max_rounds,
        "initial_packets": initial_packets,
        "packet_attempts": attempts,
        "retransmission_packets": attempts - initial_packets,
        "successful_packet_attempts": successes,
        "undetected_error_packets": undetected_packets,
        "transmissions_per_packet": transmission_multiplier,
        "final_erased_tokens": int(missing.sum()),
        "final_erasure_rate": float(missing.mean()),
        "token_error_rate": float((restored_np != ids).mean()),
        "recovered_token_accuracy": float((restored_np == ids).mean()),
        "tce_pixels_per_token_bit": 256 * 256 / (transmission_multiplier * source_bits),
        "mean_semantic_confidence_on_erased_tokens": (
            float(assigned_confidence[torch.from_numpy(missing)].mean()) if bool(missing.any()) else None
        ),
        "rounds": round_rows,
        "snr_db": float(snr_db),
        "snr_definition": "Es/N0",
        "channel": channel,
        "packet_tokens": packet_tokens,
        "token_bits": width,
    }
    return restored_np, missing, metrics
