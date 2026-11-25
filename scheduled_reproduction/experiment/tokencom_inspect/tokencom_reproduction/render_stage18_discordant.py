"""Render only the eight development images where PSNR and feature rankings differ."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cache_real_tokens import load_image
from models import erase_packets
from pretrained_vq import PretrainedVQGAN, sha256
from run_pretrained import spatial_fill


def tile(tensor):
    pixels = (tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, 'RGB')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    root = Path(__file__).resolve().parent
    out_dir = root.parents[2] / 'outputs'
    data_dir = root / 'data/imagenette_stage5'
    cases = json.loads((root / 'runs/stage18_cross_metric/results.json').read_text(encoding='utf-8'))['discordant_rows']
    assert len(cases) == 8
    torch.set_num_threads(4)
    val = torch.load(data_dir / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    with torch.random.fork_rng():
        torch.manual_seed(20261021)
        missing = erase_packets(val['ids'], .3, 4)
    cfg = maskgit_class_cond_config.get_config()
    model = BidirectionalTransformer(
        num_image_tokens=257, num_codebook_vectors=2025,
        dim=cfg.transformer.num_embeds, n_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.intermediate_size,
        num_heads=cfg.transformer.num_heads,
        attention_dropout=cfg.transformer.dropout_rate,
        hidden_dropout=cfg.transformer.dropout_rate).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                     weights_only=True)['state_dict'], strict=True)
    tokenizer = PretrainedVQGAN().eval()
    canvas = Image.new('RGB', (3 * 256, 8 * 286 + 32), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(('Original', 'Pretrained MaskGIT', 'Spatial filling')):
        draw.text((col * 256 + 5, 8), title, fill='black')
    with torch.inference_mode():
        for line, case in enumerate(cases):
            index = case['source_index']
            target = val['ids'][index]
            mask = missing[index]
            record = val['records'][index]
            path = data_dir / record['image']
            assert sha256(path) == record['sha256'] == case['image_sha256']
            original = load_image(path)
            received = target.masked_fill(mask, 2024)
            prediction = model(torch.cat((torch.tensor([2024]), received))[None])[:, 1:, :1024].argmax(-1)[0]
            pretrained = torch.where(mask, prediction, target)
            spatial = torch.from_numpy(spatial_fill(
                target.masked_fill(mask, 0)[None].numpy(), mask[None].numpy())[0]).long()
            recon = tokenizer.decode(torch.stack((pretrained, spatial)))
            y = 32 + line * 286
            for col, im in enumerate((original, recon[0], recon[1])):
                canvas.paste(tile(im), (col * 256, y))
            label = (f"Index {index} | PSNR delta {case['psnr_difference_db']:+.2f} dB | "
                     f"ResNet cosine delta {case['feature_cosine_difference']:+.3f}")
            draw.text((5, y + 258), label, fill='black')
    output = out_dir / 'TokCom_stage18_metric_discordance_montage.png'
    canvas.save(output)
    print(output)


if __name__ == '__main__': main()
