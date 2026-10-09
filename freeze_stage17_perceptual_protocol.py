"""Freeze the Stage-17 perceptual-evaluation protocol after Stage 16."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE17_PERCEPTUAL_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_selective(summary: list[dict]) -> tuple[str, dict]:
    scores = {}
    for name in ("selective_001", "selective_002", "selective_004"):
        channel_scores = []
        for channel in ("awgn", "rayleigh"):
            rows = {row["policy_name"]: row for row in summary if row["channel"] == channel}
            lower, upper, candidate = rows["semantic_only"], rows["full_arq"], rows[name]
            accuracy = (candidate["recovered_token_accuracy_mean"] - lower["recovered_token_accuracy_mean"]) / (upper["recovered_token_accuracy_mean"] - lower["recovered_token_accuracy_mean"])
            efficiency = (candidate["tce_mean"] - upper["tce_mean"]) / (lower["tce_mean"] - upper["tce_mean"])
            harmonic = 2 * accuracy * efficiency / (accuracy + efficiency)
            channel_scores.append({"channel": channel, "accuracy_fraction": accuracy, "tce_fraction": efficiency, "harmonic": harmonic})
        scores[name] = {"channels": channel_scores, "mean_harmonic": sum(row["harmonic"] for row in channel_scores) / len(channel_scores)}
    selected = max(scores, key=lambda name: scores[name]["mean_harmonic"])
    return selected, scores


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-17 protocol is already frozen")
    stage16 = json.loads((ROOT / "STAGE16_ARQ_TCE_SUMMARY.json").read_text(encoding="utf-8"))
    status = json.loads((ROOT / "runs/stage16_arq_tce/status.json").read_text(encoding="utf-8"))
    if stage16.get("status") != "completed" or not status.get("completed") or stage16["conditions"] != 30:
        raise ValueError("Stage 16 is incomplete")
    if stage16.get("test_accessed") or stage16.get("stage14_holdout_used"):
        raise ValueError("Stage-16 data isolation failed")
    selected, selection_scores = choose_selective(stage16["summary"])
    policies = {
        "selective_001": 0.01,
        "selective_002": 0.02,
        "selective_004": 0.04,
    }
    modes = {
        "no_cmi": {"context": "none", "policy": "confidence", "confidence_threshold": 0.0, "guidance_scale": 0.0},
        "cmi_no_arq": {"context": "class", "policy": "confidence", "confidence_threshold": 0.0, "guidance_scale": 1.0},
        "cmi_selective": {"context": "class", "policy": "confidence", "confidence_threshold": policies[selected], "guidance_scale": 1.0},
        "full_arq": {"context": "class", "policy": "all_failed", "confidence_threshold": 1.0, "guidance_scale": 1.0},
    }
    conditions = []
    for channel, snr in (("awgn", 5.0), ("rayleigh", 10.0)):
        for seed in (16101, 16102, 16103):
            for mode in modes:
                conditions.append({"channel": channel, "snr_esn0_db": snr, "seed": seed, "mode": mode})
    implementation = [
        "evaluate_stage17_perceptual.py",
        "tokcom_image_metrics.py",
        "pretrained_vq.py",
        "tokcom_arq.py",
        "context_maskgit_v3.py",
        "digital_link.py",
    ]
    stage16_protocol = json.loads((ROOT / "STAGE16_ARQ_TCE_PROTOCOL.json").read_text(encoding="utf-8"))
    protocol = {
        "stage": 17,
        "status": "frozen",
        "purpose": "TokCom PSNR, LPIPS, CLIP, TER and TCE comparison on development data",
        "evidence_level": "development evaluation; not a new blind test",
        "implementation_sha256": {name: sha256(ROOT / name) for name in implementation},
        "checkpoint": stage16_protocol["checkpoint"],
        "checkpoint_sha256": stage16_protocol["checkpoint_sha256"],
        "validation_cache": stage16_protocol["validation_cache"],
        "validation_cache_sha256": stage16_protocol["validation_cache_sha256"],
        "dataset_root": "data/imagenette_stage10",
        "clip_checkpoint": "pretrained/clip-vit-base-patch16-pytorch_model.bin",
        "clip_checkpoint_sha256": sha256(ROOT / "pretrained/clip-vit-base-patch16-pytorch_model.bin"),
        "vqgan_checkpoint": "pretrained/maskgit-vqgan-imagenet-f16-256.bin",
        "vqgan_checkpoint_sha256": sha256(ROOT / "pretrained/maskgit-vqgan-imagenet-f16-256.bin"),
        "images": 150,
        "max_rounds": 4,
        "recovery_steps": 8,
        "selected_stage16_policy": selected,
        "selection_rule": "maximum mean harmonic balance of normalized token-accuracy gain and retained TCE across both channels",
        "selection_scores": selection_scores,
        "modes": modes,
        "conditions": conditions,
        "preview_indices": [0, 15, 30, 45, 60, 75, 90, 105, 120, 135],
        "test_accessed": False,
        "stage14_holdout_used": False,
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(OUTPUT), "conditions": len(conditions), "selected": selected, "sha256": sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
