"""Compare current VQGAN IDs with the independently converted MaskGIT tokenizer.

This is a development-only check. It does not touch the held-out test set.
The third-party reference is https://github.com/hmorimitsu/maskgit-torch .
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pretrained_vq import DEFAULT_WEIGHT, PretrainedVQGAN, sha256


def image_tensor(path):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    scale = 256 / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (image.width - 256) // 2
    top = (image.height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=Path('data/public_samples/dog.jpg'))
    parser.add_argument('--val-cache', type=Path,
                        help='Optional stage5 val.pt: add the first image of each class')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import vqgan_config
    from maskgit.nets.vqgan_tokenizer import VQVAE

    torch.set_num_threads(4)
    current = PretrainedVQGAN()
    saved = torch.load(args.candidate, map_location='cpu', weights_only=True)
    candidate = VQVAE(vqgan_config.get_config()).eval()
    candidate.load_state_dict(saved['state_dict'], strict=True)
    candidate.requires_grad_(False)

    current_book = current.quantize.embedding.weight.detach()
    candidate_book = candidate.quantizer.codebook.weight.detach()
    book_difference = (current_book - candidate_book).abs()
    images = [args.image]
    if args.val_cache:
        cache = torch.load(args.val_cache, map_location='cpu', weights_only=True)
        seen = set()
        for record in cache['records']:
            if record['label'] not in seen:
                seen.add(record['label'])
                images.append(args.val_cache.parent.parent / record['image'])
    matches = 0
    count = 0
    max_decode_difference = 0.
    mean_decode_difference = []
    with torch.no_grad():
        for path in images:
            image = image_tensor(path)
            current_ids = current.encode(image).reshape(-1)
            candidate_ids = candidate.encode_to_indices({'image': image}).reshape(-1)
            matches += int((current_ids == candidate_ids).sum())
            count += current_ids.numel()
            current_decode = current.decode(current_ids.reshape(1, -1))
            candidate_decode = candidate.decode_from_indices({
                'encoding_indices': candidate_ids.reshape(1, 16, 16)}).clamp(0, 1)
            difference = (current_decode - candidate_decode).abs()
            max_decode_difference = max(max_decode_difference, float(difference.max()))
            mean_decode_difference.append(float(difference.mean()))
    result = {
        'scope': 'public sample and first stage5 validation image per class; no stage7 holdout access',
        'current_sha256': sha256(DEFAULT_WEIGHT),
        'candidate_sha256': sha256(args.candidate),
        'image_sha256': [sha256(path) for path in images],
        'images': len(images),
        'codebook_shape': list(current_book.shape),
        'codebook_exact_equal': bool(torch.equal(current_book, candidate_book)),
        'codebook_max_abs_difference': float(book_difference.max()),
        'token_count': count,
        'token_id_exact_matches': matches,
        'token_ids_exact_equal': matches == count,
        'decode_max_abs_difference': max_decode_difference,
        'decode_mean_abs_difference': sum(mean_decode_difference) / len(mean_decode_difference),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
