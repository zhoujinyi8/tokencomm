"""Shared deterministic split and selection rules for Stage 23."""
from __future__ import annotations

import hashlib


def ranked_class_indices(records: list[dict], label: int, seed: int) -> list[int]:
    candidates = [(index, row) for index, row in enumerate(records) if int(row["label"]) == label]
    return [
        index
        for index, _ in sorted(
            candidates,
            key=lambda item: hashlib.sha256(f"{seed}:{item[1]['image']}".encode()).digest(),
        )
    ]


def balanced_window_indices(
    records: list[dict], *, start_per_class: int, images_per_class: int, seed: int, classes: int = 101
) -> list[int]:
    """Select a stable balanced window from each class-specific hash ranking."""
    if start_per_class < 0 or images_per_class < 1:
        raise ValueError("Balanced-window bounds must be positive")
    selected: list[int] = []
    stop = start_per_class + images_per_class
    for label in range(classes):
        ranked = ranked_class_indices(records, label, seed)
        if len(ranked) < stop:
            raise ValueError(f"Class {label} lacks enough development records")
        selected.extend(ranked[start_per_class:stop])
    if len(selected) != classes * images_per_class or len(set(selected)) != len(selected):
        raise ValueError("Balanced-window selection is not unique and complete")
    return selected


def select_threshold(rows: list[dict], thresholds: list[float], recovery_fraction: float) -> dict:
    """Pick the most efficient threshold reaching a fixed fraction of full-ARQ recovery.

    The rule is applied to one channel after averaging the frozen calibration seeds.
    A candidate qualifies when it closes at least ``recovery_fraction`` of the token
    error-rate gap between semantic-only transmission and full ARQ. Among qualifying
    candidates, maximum TCE wins; ties use lower token error and then lower threshold.
    """
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("Recovery fraction must be in (0,1]")
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    if "semantic_only" not in by_mode or "full_arq" not in by_mode:
        raise ValueError("Calibration rows require semantic-only and full-ARQ anchors")

    def mean(mode: str, metric: str) -> float:
        values = [float(row["metrics"][metric]) for row in by_mode.get(mode, [])]
        if not values:
            raise ValueError(f"Missing calibration mode: {mode}")
        return sum(values) / len(values)

    semantic_error = mean("semantic_only", "token_error_rate")
    full_error = mean("full_arq", "token_error_rate")
    target_error = semantic_error - recovery_fraction * max(0.0, semantic_error - full_error)
    candidates = []
    for threshold in thresholds:
        mode = f"selective_{threshold:g}"
        error = mean(mode, "token_error_rate")
        tce = mean(mode, "tce_pixels_per_token_bit")
        candidates.append({"threshold": float(threshold), "token_error_rate": error, "tce": tce})
    qualifying = [row for row in candidates if row["token_error_rate"] <= target_error]
    if qualifying:
        selected = min(qualifying, key=lambda row: (-row["tce"], row["token_error_rate"], row["threshold"]))
        fallback = False
    else:
        selected = min(candidates, key=lambda row: (row["token_error_rate"], -row["tce"], row["threshold"]))
        fallback = True
    return {
        **selected,
        "semantic_only_token_error_rate": semantic_error,
        "full_arq_token_error_rate": full_error,
        "target_token_error_rate": target_error,
        "recovery_fraction": recovery_fraction,
        "fallback_used": fallback,
    }
