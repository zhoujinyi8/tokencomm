"""Compare the author's VQGAN checkpoint with our current tokenizer on dev images."""
import argparse
import json
import sys
from pathlib import Path

import torch

from cache_real_tokens import load_image
from pretrained_vq import PretrainedVQGAN, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('pretrained/official_tokcom/last.ckpt'))
    parser.add_argument('--data', type=Path, default=Path('data/imagenette_stage5'))
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from omegaconf import OmegaConf
    from Network.Taming.models.vqgan import VQModel

    torch.set_num_threads(4)
    config = OmegaConf.load(args.official_source / 'pretrained_maskgit/VQGAN/model.yaml')
    official = VQModel(**config.model.params).eval()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    incompatible = official.load_state_dict(state, strict=False)
    important = ('encoder.', 'decoder.', 'quantize.', 'quant_conv.', 'post_quant_conv.')
    missing_important = [key for key in incompatible.missing_keys if key.startswith(important)]
    if missing_important:
        raise ValueError(f'Official checkpoint missing VQGAN parameters: {missing_important[:10]}')
    del saved, state
    official.requires_grad_(False)
    current = PretrainedVQGAN().eval()
    codebook_official = official.quantize.embedding.weight.detach()
    codebook_current = current.quantize.embedding.weight.detach()
    cache = torch.load(args.data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    records = cache['records'][:args.limit]
    rows = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            path = args.data / record['image']
            if sha256(path) != record['sha256']:
                raise ValueError(f'Image changed: {path}')
            image = load_image(path)[None]
            _, _, (_, _, ids) = official.encode(image)
            official_ids = ids.reshape(1, 256)
            current_ids = current.encode(image).reshape(1, 256)
            official_image = official.decode_code(official_ids.reshape(1, 16, 16)).clamp(0, 1)
            current_image = current.decode(current_ids)
            rows.append({
                'source_index': index, 'image_sha256': record['sha256'],
                'matching_token_ids': int((official_ids == current_ids).sum()),
                'official_token_min': int(official_ids.min()),
                'official_token_max': int(official_ids.max()),
                'mean_abs_reconstruction_difference': float((official_image - current_image).abs().mean()),
                'max_abs_reconstruction_difference': float((official_image - current_image).abs().max()),
            })
            print(json.dumps(rows[-1]), flush=True)
    result = {
        'scope': 'stage5 dev first cached images; one common image tensor per paired comparison',
        'official_source_commit': '0422ed32dcde3347e0bb6697f343f7a9ff7b3214',
        'official_vqgan_checkpoint_sha256': sha256(args.checkpoint),
        'current_vqgan_checkpoint_sha256': sha256(Path('pretrained/maskgit-vqgan-imagenet-f16-256.bin')),
        'official_checkpoint_unused_keys_count': len(incompatible.unexpected_keys),
        'official_checkpoint_missing_keys': incompatible.missing_keys,
        'codebook_exact_equal': bool(torch.equal(codebook_official, codebook_current)),
        'codebook_max_abs_difference': float((codebook_official-codebook_current).abs().max()),
        'images': len(rows), 'matching_token_ids_total': sum(row['matching_token_ids'] for row in rows),
        'token_ids_total': 256 * len(rows), 'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('images', 'matching_token_ids_total',
                                                   'token_ids_total', 'codebook_exact_equal')}), flush=True)


if __name__ == '__main__':
    main()
