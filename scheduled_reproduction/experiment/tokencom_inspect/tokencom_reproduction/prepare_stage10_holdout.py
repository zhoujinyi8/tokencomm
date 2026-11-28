"""Prepare a new unopened Imagenette holdout, excluding every earlier split.

This creates a manifest and source images only. It never encodes or evaluates them.
Do not run it until the receiver rule has been fixed from validation data.
"""
import argparse
import json
from pathlib import Path

from prepare_stage7_holdout import sample
from pretrained_vq import sha256


def prior_hashes(datasets):
    file_hashes, pixel_hashes = set(), set()
    for folder in datasets:
        for split in ('train', 'val', 'test'):
            path = folder / f'{split}.jsonl'
            if not path.exists():
                continue
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                file_hashes.add(row['sha256'])
                if 'pixel_sha256' in row:
                    pixel_hashes.add(row['pixel_sha256'])
    return file_hashes, pixel_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-parquet', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/imagenette_stage10_holdout'))
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty output directory')
    prior = [Path('data/imagenette_pilot'), Path('data/imagenette_stage5'),
             Path('data/imagenette_stage7_holdout')]
    excluded_files, excluded_pixels = prior_hashes(prior)
    selected, names, eligible, rejected = sample(
        args.validation_parquet, args.per_class, args.seed, excluded_files, excluded_pixels)
    records = []
    selected_files, selected_pixels = set(), set()
    for label, group in enumerate(selected):
        folder = args.out / 'images' / 'test' / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        for raw, digest, pixels, source_path in group:
            if digest in selected_files or pixels in selected_pixels:
                raise ValueError('Duplicate within new holdout')
            selected_files.add(digest)
            selected_pixels.add(pixels)
            path = folder / (digest[:20] + '.jpg')
            path.write_bytes(raw)
            records.append({
                'image': path.relative_to(args.out).as_posix(), 'label': label,
                'question': 'What is the main object in this image?',
                'answer': names[label].lower(), 'sha256': digest,
                'pixel_sha256': pixels, 'source_path': source_path,
            })
    (args.out / 'test.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
        encoding='utf-8')
    manifest = {
        'dataset': 'Imagenette 320px',
        'purpose': 'new holdout for frozen stage10 pretrained receiver',
        'source_dataset': 'https://huggingface.co/datasets/ShaomuTan/imagenette',
        'source_parquet_sha256': sha256(args.validation_parquet),
        'seed': args.seed, 'per_class': args.per_class, 'count': len(records),
        'class_names': names, 'excluded_datasets': [str(p) for p in prior],
        'excluded_prior_file_hashes': len(excluded_files),
        'excluded_prior_pixel_hashes': len(excluded_pixels),
        'source_rows_rejected': rejected, 'source_rows_eligible_by_class': eligible,
        'prior_file_overlap': 0, 'prior_pixel_overlap': 0,
        'test_policy': 'Do not encode or evaluate until receiver, mask seed, PERs, and metrics are frozen.',
    }
    (args.out / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'count': len(records), 'excluded': rejected}, ensure_ascii=False))


if __name__ == '__main__':
    main()
