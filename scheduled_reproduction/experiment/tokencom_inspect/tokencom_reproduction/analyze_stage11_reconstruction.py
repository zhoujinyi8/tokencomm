"""Image-level descriptive summary of stage11 validation reconstruction."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Output file already exists')
    partial = args.input.suffix == '.jsonl'
    if partial:
        rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines()
                if line.strip()]
        protocol = json.loads(args.input.with_name(
            args.input.name.replace('.progress.jsonl', '.protocol.json')).read_text(encoding='utf-8'))
        if [r['source_index'] for r in rows] != protocol['selected_indices'][:len(rows)]:
            raise ValueError('Progress is not a protocol prefix')
        if not rows: raise ValueError('No completed image rows')
        means = lambda key: {name: sum(r[key][name] for r in rows) / len(rows)
                             for name in rows[0][key]}
        data = {'rows': rows, 'images': len(rows), 'per': protocol['per'],
                'seed': protocol['seed'],
                'mean_psnr_to_original_db': means('psnr_to_original_db'),
                'mean_psnr_to_clean_vqgan_db': means('psnr_to_clean_vqgan_db')}
    else:
        data = json.loads(args.input.read_text(encoding='utf-8'))
    rows = data['rows']
    if data['images'] != len(rows) or len({r['image_sha256'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate image rows')
    methods = ['pretrained_masked_class', 'stage7_one_pass', 'spatial']
    delta = np.array([r['psnr_to_original_db']['pretrained_masked_class'] -
                      r['psnr_to_original_db']['spatial'] for r in rows])
    rng = np.random.default_rng(20261027)
    draws = delta[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(axis=1)
    by_class = defaultdict(list)
    for r, d in zip(rows, delta): by_class[r['class']].append(float(d))
    total_missing = sum(r['missing_tokens'] for r in rows)
    result = {
        'scope': ('exploratory stage5 validation; image-level paired descriptive bootstrap'
                  + ('; incomplete ordered prefix' if partial else '')),
        'images': len(rows), 'per': data['per'], 'seed': data['seed'],
        'mean_psnr_to_original_db': data['mean_psnr_to_original_db'],
        'mean_psnr_to_clean_vqgan_db': data['mean_psnr_to_clean_vqgan_db'],
        'pretrained_minus_spatial_mean_db': float(delta.mean()),
        'pretrained_minus_spatial_bootstrap_95pct_db':
            (None if partial else [float(v) for v in np.quantile(draws, [.025, .975])]),
        'pretrained_better_than_spatial_images': int((delta > 0).sum()),
        'spatial_better_than_pretrained_images': int((delta < 0).sum()),
        'class_mean_pretrained_minus_spatial_db':
            {name: float(np.mean(values)) for name, values in sorted(by_class.items())},
        'erased_token_accuracy_selected_images': {
            name: sum(r['token_hits'][name] for r in rows) / total_missing for name in methods},
        'note': ('Ordered partial prefix is not representative; no interval reported.' if partial else
                 'Bootstrap interval is exploratory validation evidence, not a held-out significance claim.')
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
