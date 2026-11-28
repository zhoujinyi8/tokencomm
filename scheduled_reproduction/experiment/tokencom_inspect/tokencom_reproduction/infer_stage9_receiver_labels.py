"""Infer ImageNet class from only received tokens via spatial fill + ResNet18.

This is a receiver-side auxiliary model; it consumes no ground-truth class label.
"""
import argparse
import json
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

from cache_pretrained_vision import load_backbone
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS
from models import erase_packets
from pretrained_vq import PretrainedVQGAN, sha256
from run_pretrained import spatial_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/imagenette_stage5/token_cache/val.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--per', type=float, default=.3)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--all-images', action='store_true',
                        help='Use every stage5 validation image instead of first per class')
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output file already exists')
    torch.set_num_threads(4)
    val = torch.load(args.data, map_location='cpu', weights_only=True)
    if args.all_images:
        indices = list(range(len(val['ids'])))
    else:
        indices = []
        seen = set()
        for index, label in enumerate(val['labels'].tolist()):
            if label not in seen:
                indices.append(index)
                seen.add(label)
            if len(indices) >= args.limit: break
    ids = val['ids'][indices]
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, args.per, 4)
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
    vq = PretrainedVQGAN()
    classifier, classifier_digest = load_backbone()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rows = []
    with torch.inference_mode():
        for row, index in enumerate(indices):
            image = vq.decode(spatial[row:row + 1])[0]
            predicted_class = int(classifier(transform(image).unsqueeze(0)).argmax())
            true_class = IMAGENET_CLASS_IDS[val['class_names'][int(val['labels'][index])]]
            rows.append({'source_index': index, 'predicted_class': predicted_class,
                         'true_class': true_class, 'correct': predicted_class == true_class})
            print(json.dumps(rows[-1]), flush=True)
    result = {
        'split': 'stage5 validation only', 'test_accessed': False,
        'method': 'spatial-filled received VQGAN image -> torchvision ResNet18 IMAGENET1K_V1',
        'data_identity': val['identity'], 'classifier_sha256': classifier_digest,
        'tokenizer_sha256': sha256(Path(__file__).parent / 'pretrained/maskgit-vqgan-imagenet-f16-256.bin'),
        'seed': args.seed, 'per': args.per, 'rows': rows,
        'selection': 'all validation images' if args.all_images else 'first validation image per class',
        'class_accuracy': sum(row['correct'] for row in rows) / len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
