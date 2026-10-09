"""Image-quality metrics used by the TokCom paper evaluation.

The module deliberately keeps model loading separate from metric computation so
large public backbones are loaded once and reused over many decoded batches.
CLIP score is the cosine similarity between original and reconstructed image
embeddings. LPIPS uses the pinned official implementation in ``third_party``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parent
LPIPS_ROOT = ROOT / "third_party/perceptual_similarity"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_images(original: torch.Tensor, reconstructed: torch.Tensor) -> None:
    if original.shape != reconstructed.shape or original.ndim != 4 or original.shape[1] != 3:
        raise ValueError("Expected matching [B,3,H,W] image tensors")
    if not torch.isfinite(original).all() or not torch.isfinite(reconstructed).all():
        raise ValueError("Images must contain only finite values")
    if original.min() < 0 or original.max() > 1 or reconstructed.min() < 0 or reconstructed.max() > 1:
        raise ValueError("Images must be in [0,1]")


def psnr_per_image(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Return RGB PSNR for each image, assuming a data range of one."""
    _validate_images(original, reconstructed)
    mse = (original - reconstructed).square().flatten(1).mean(1)
    return torch.where(mse == 0, torch.full_like(mse, torch.inf), -10.0 * torch.log10(mse))


class ClipImageSimilarity(nn.Module):
    """Pinned CLIP ViT-B/16 image-to-image cosine similarity backend.

    The Hugging Face checkpoint contains a full CLIP model. Only the vision
    tower and visual projection are instantiated, which reduces memory while
    preserving the published image embedding exactly.
    """

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
        config = CLIPVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            projection_dim=512,
        )
        self.model = CLIPVisionModelWithProjection(config)
        full_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        vision_state = {
            name: value
            for name, value in full_state.items()
            if (name.startswith("vision_model.") or name.startswith("visual_projection."))
            and not name.endswith("position_ids")
        }
        self.model.load_state_dict(vision_state, strict=True)
        self.model.requires_grad_(False).eval()
        self.checkpoint = str(checkpoint.resolve())
        self.checkpoint_sha256 = sha256(checkpoint)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        images = (images - self.mean) / self.std
        features = self.model(pixel_values=images).image_embeds
        return F.normalize(features, dim=-1)

    @torch.inference_mode()
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        _validate_images(original, reconstructed)
        return (self.embed(original) * self.embed(reconstructed)).sum(-1)


class LpipsSqueezeSimilarity(nn.Module):
    """Official LPIPS v0.1 metric with its calibrated SqueezeNet backbone."""

    def __init__(self):
        super().__init__()
        path = str(LPIPS_ROOT)
        if path not in sys.path:
            sys.path.insert(0, path)
        import lpips as official_lpips

        module_path = Path(official_lpips.__file__).resolve()
        if LPIPS_ROOT.resolve() not in module_path.parents:
            raise RuntimeError(f"Unpinned LPIPS implementation imported from {module_path}")
        self.model = official_lpips.LPIPS(net="squeeze", version="0.1", verbose=False)
        self.model.requires_grad_(False).eval()
        self.linear_weight = LPIPS_ROOT / "lpips/weights/v0.1/squeeze.pth"
        self.linear_weight_sha256 = sha256(self.linear_weight)
        self.trunk_weight = Path(torch.hub.get_dir()) / "checkpoints/squeezenet1_1-b8a52dc0.pth"
        if not self.trunk_weight.is_file():
            raise FileNotFoundError("torchvision did not materialize the SqueezeNet checkpoint")
        self.trunk_weight_sha256 = sha256(self.trunk_weight)
        if self.trunk_weight_sha256 != "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b":
            raise ValueError("Unexpected SqueezeNet ImageNet checkpoint hash")

    @torch.inference_mode()
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        _validate_images(original, reconstructed)
        return self.model(original, reconstructed, normalize=True).flatten()


class TokComImageMetrics:
    """Compute PSNR, CLIP cosine similarity, and LPIPS in bounded batches."""

    def __init__(self, clip_checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.clip = ClipImageSimilarity(clip_checkpoint).to(self.device)
        self.lpips = LpipsSqueezeSimilarity().to(self.device)

    @torch.inference_mode()
    def evaluate(self, original: torch.Tensor, reconstructed: torch.Tensor, batch_size: int = 8) -> dict:
        _validate_images(original, reconstructed)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = {"psnr": [], "clip_cosine": [], "lpips": []}
        for start in range(0, len(original), batch_size):
            target = original[start : start + batch_size].to(self.device)
            result = reconstructed[start : start + batch_size].to(self.device)
            values["psnr"].append(psnr_per_image(target, result).cpu())
            values["clip_cosine"].append(self.clip(target, result).cpu())
            values["lpips"].append(self.lpips(target, result).cpu())
        arrays = {name: torch.cat(parts) for name, parts in values.items()}
        return {
            "per_image": {name: value.tolist() for name, value in arrays.items()},
            "mean": {name: float(value.mean()) for name, value in arrays.items()},
            "images": len(original),
            "definition": {
                "psnr": "RGB PSNR, data_range=1, higher is better",
                "clip_cosine": "CLIP ViT-B/16 original/reconstruction image embedding cosine, higher is better",
                "lpips": "LPIPS v0.1 calibrated SqueezeNet distance, lower is better",
            },
            "provenance": {
                "clip_checkpoint": self.clip.checkpoint,
                "clip_checkpoint_sha256": self.clip.checkpoint_sha256,
                "lpips_upstream_commit": "082bb24f84c091ea94de2867d34c4544f68e0963",
                "lpips_linear_weight_sha256": self.lpips.linear_weight_sha256,
                "lpips_squeezenet_weight_sha256": self.lpips.trunk_weight_sha256,
            },
        }
