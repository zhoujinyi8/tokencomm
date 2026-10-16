"""Audit completed Stage-19 results and produce compact machine artifacts/figures."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from evaluate_stage19_v4_link import condition_key


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/stage19_v4_link"
FIGURES = ROOT / "figures"
MODES = ("no_cmi", "cmi_no_arq", "cmi_selective", "full_arq")
MODE_LABELS = ("No CMI", "CMI, no ARQ", "CMI, selective ARQ", "Full ARQ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(protocol: dict) -> tuple[list[dict], dict]:
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    if status != {"completed": True, "conditions": 24, "test_accessed": False, "stage14_holdout_used": False}:
        raise ValueError("Stage-19 status is incomplete or violates isolation")
    expected = {condition_key(condition) for condition in protocol["conditions"]}
    if len(expected) != 24:
        raise ValueError("Stage-19 protocol condition keys are not unique")
    records = []
    per_image_paths = []
    preview_count = 0
    reference_paths = sorted((RUN / "reference").glob("[0-9][0-9][0-9][0-9].pt"))
    if len(reference_paths) != protocol["images"]:
        raise ValueError("Reference cache is incomplete")
    for condition in protocol["conditions"]:
        key = condition_key(condition)
        directory = RUN / key
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["condition"] != condition or summary["images"] != protocol["images"]:
            raise ValueError(f"Condition identity/count mismatch: {key}")
        if summary.get("test_accessed") or summary.get("stage14_holdout_used"):
            raise ValueError(f"Data-isolation violation: {key}")
        rows = []
        for index in range(protocol["images"]):
            path = directory / "per_image" / f"{index:04d}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["index"] != index:
                raise ValueError(f"Per-image index mismatch: {key}/{index}")
            for name in ("token_accuracy", "psnr", "clip_cosine", "lpips", "vqgan_baseline_psnr"):
                if not math.isfinite(float(row[name])):
                    raise ValueError(f"Non-finite {name}: {key}/{index}")
            rows.append(row)
            per_image_paths.append(path)
        for name in ("token_accuracy", "psnr", "clip_cosine", "lpips", "vqgan_baseline_psnr"):
            measured = float(np.mean([row[name] for row in rows]))
            if not math.isclose(measured, summary["mean"][name], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Aggregate mismatch for {name}: {key}")
        communication_accuracy = summary["communication_metrics"]["recovered_token_accuracy"]
        if not math.isclose(summary["mean"]["token_accuracy"], communication_accuracy, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Token accuracy mismatch: {key}")
        previews = list((directory / "previews").glob("*.png"))
        if len(previews) != len(protocol["preview_indices"]):
            raise ValueError(f"Preview count mismatch: {key}")
        preview_count += len(previews)
        records.append(summary)
    baseline_values = [row["mean"]["vqgan_baseline_psnr"] for row in records]
    if max(baseline_values) - min(baseline_values) > 1e-10:
        raise ValueError("VQGAN reference baseline changed between conditions")
    audit_result = {
        "status": "passed",
        "conditions": len(records),
        "images_per_condition": protocol["images"],
        "per_image_rows": len(per_image_paths),
        "reference_images": len(reference_paths),
        "previews": preview_count,
        "condition_keys_unique": True,
        "metrics_finite": True,
        "aggregates_recomputed": True,
        "communication_accuracy_matches_per_image": True,
        "vqgan_baseline_constant": True,
        "test_accessed": False,
        "stage14_holdout_used": False,
        "protocol_sha256": sha256(ROOT / "STAGE19_V4_LINK_PROTOCOL.json"),
        "aggregate_summary_sha256": sha256(RUN / "summary.json"),
        "per_image_json_combined_sha256": combined_sha256(per_image_paths),
    }
    return records, audit_result


def plot_metrics(summary: list[dict]) -> None:
    FIGURES.mkdir(exist_ok=True)
    specs = (
        ("token_accuracy_mean", "Token accuracy", 100.0),
        ("psnr_mean", "PSNR (dB)", 1.0),
        ("clip_cosine_mean", "CLIP cosine", 1.0),
        ("lpips_mean", "LPIPS", 1.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(MODES))
    width = 0.36
    for axis, (field, title, scale) in zip(axes.flat, specs):
        for offset, channel in ((-width / 2, "awgn"), (width / 2, "rayleigh")):
            rows = {row["mode"]: row for row in summary if row["channel"] == channel}
            values = [rows[mode][field] * scale for mode in MODES]
            axis.bar(x + offset, values, width, label=channel.upper())
        axis.set_title(title)
        axis.set_xticks(x, MODE_LABELS, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Stage 19 TokCom perceptual evaluation (development set)")
    figure.savefig(FIGURES / "stage19_perceptual_metrics.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for channel, marker in (("awgn", "o"), ("rayleigh", "s")):
        rows = {row["mode"]: row for row in summary if row["channel"] == channel}
        tce = [rows[mode]["tce_mean"] for mode in MODES]
        clip = [rows[mode]["clip_cosine_mean"] for mode in MODES]
        lpips = [rows[mode]["lpips_mean"] for mode in MODES]
        axes[0].plot(tce, clip, marker=marker, label=channel.upper())
        axes[1].plot(tce, lpips, marker=marker, label=channel.upper())
        for index, label in enumerate(("N", "C", "S", "F")):
            axes[0].annotate(label, (tce[index], clip[index]))
            axes[1].annotate(label, (tce[index], lpips[index]))
    axes[0].set(xlabel="TCE", ylabel="CLIP cosine", title="Semantic quality vs efficiency")
    axes[1].set(xlabel="TCE", ylabel="LPIPS", title="Perceptual distance vs efficiency")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(FIGURES / "stage19_tce_quality_tradeoff.png", dpi=180)
    plt.close(figure)


def make_comparison(protocol: dict, image_index: int = 0, seed: int = 16101) -> None:
    columns = ("Original", "VQGAN", *MODE_LABELS)
    cell = 256
    header = 38
    canvas = Image.new("RGB", (cell * len(columns), header + cell * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * cell + 8, 12), title, fill="black")
    for row_index, (channel, snr) in enumerate((("awgn", 5.0), ("rayleigh", 10.0))):
        base_y = header + row_index * cell
        for mode_index, mode in enumerate(MODES):
            condition = {"channel": channel, "snr_esn0_db": snr, "seed": seed, "mode": mode}
            preview = Image.open(RUN / condition_key(condition) / "previews" / f"{image_index:04d}_original_vqgan_received.png").convert("RGB")
            if mode_index == 0:
                canvas.paste(preview.crop((0, 0, cell, cell)), (0, base_y))
                canvas.paste(preview.crop((cell, 0, cell * 2, cell)), (cell, base_y))
            reconstructed = preview.crop((cell * 2, 0, cell * 3, cell))
            canvas.paste(reconstructed, ((mode_index + 2) * cell, base_y))
        draw.text((8, base_y + 8), channel.upper(), fill="white", stroke_fill="black", stroke_width=2)
    canvas.save(FIGURES / "stage19_mode_comparison.png")


def main() -> None:
    protocol = json.loads((ROOT / "STAGE19_V4_LINK_PROTOCOL.json").read_text(encoding="utf-8"))
    records, audit_result = audit(protocol)
    aggregate = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    stage17 = json.loads((ROOT / "STAGE17_PERCEPTUAL_SUMMARY.json").read_text(encoding="utf-8"))
    v3_rows = {(row["channel"], row["mode"]): row for row in stage17["summary"]}
    v3_to_v4_delta = []
    for row in aggregate["summary"]:
        baseline = v3_rows[(row["channel"], row["mode"])]
        v3_to_v4_delta.append({
            "channel": row["channel"],
            "mode": row["mode"],
            **{
                f"{field}_delta": float(row[field] - baseline[field])
                for field in ("token_accuracy_mean", "psnr_mean", "clip_cosine_mean", "lpips_mean", "tce_mean")
            },
        })
    plot_metrics(aggregate["summary"])
    make_comparison(protocol)
    output = {
        "stage": 19,
        "status": "completed",
        "evidence_level": protocol["evidence_level"],
        "selected_stage16_policy": protocol["selected_stage16_policy"],
        "conditions": aggregate["conditions"],
        "images_per_condition": protocol["images"],
        "summary": aggregate["summary"],
        "stage17_v3_protocol_sha256": protocol["stage17_protocol_sha256"],
        "stage17_v3_summary_sha256": protocol["stage17_summary_sha256"],
        "v3_to_v4_delta": v3_to_v4_delta,
        "audit": audit_result,
        "figures": [
            "figures/stage19_perceptual_metrics.png",
            "figures/stage19_tce_quality_tradeoff.png",
            "figures/stage19_mode_comparison.png",
        ],
    }
    atomic_json(ROOT / "STAGE19_V4_LINK_SUMMARY.json", output)
    atomic_json(ROOT / "STAGE19_AUDIT.json", audit_result)
    print(json.dumps(audit_result))


if __name__ == "__main__":
    main()

