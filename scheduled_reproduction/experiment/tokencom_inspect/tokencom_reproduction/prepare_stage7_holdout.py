"""Sample a fresh Imagenette holdout after excluding all earlier project splits.

Preparation saves images and manifests. It does not encode images or evaluate a
model, so the holdout remains unopened to model selection until final testing.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from prepare_stage5_imagenette import pixel_digest
from pretrained_vq import sha256


def prior_hashes(datasets):
    file_hashes, pixel_hashes = set(), set()
    for folder in datasets:
        for split in ('train', 'val', 'test'):
            path = folder / (split + '.jsonl')
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                file_hashes.add(row['sha256'])
                if 'pixel_sha256' in row:
                    pixel_hashes.add(row['pixel_sha256'])
    return file_hashes, pixel_hashes


def sample(parquet, count, seed, excluded_files, excluded_pixels):
    source = pq.ParquetFile(parquet)
    metadata = json.loads(source.schema_arrow.metadata[b'huggingface'])
    names = metadata['info']['features']['label']['names']
    selected = [[] for _ in names]
    eligible = [0] * len(names)
    rejected = 0
    rng = random.Random(seed)
    for batch in source.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            raw = row['image']['bytes']
            digest = hashlib.sha256(raw).hexdigest()
            if digest in excluded_files:
                rejected += 1
                continue
            pixels = pixel_digest(raw)
            if pixels in excluded_pixels:
                rejected += 1
                continue
            label = int(row['label'])
            eligible[label] += 1
            item = (raw, digest, pixels, row['image'].get('path'))
            if len(selected[label]) < count:
                selected[label].append(item)
            else:
                slot = rng.randrange(eligible[label])
                if slot < count:
                    selected[label][slot] = item
    if any(len(group) != count for group in selected):
        raise ValueError('Source has too few eligible images in at least one class')
    return selected, names, eligible, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-parquet', required=True)
    parser.add_argument('--pilot', default='data/imagenette_pilot')
    parser.add_argument('--stage5', default='data/imagenette_stage5')
    parser.add_argument('--out', default='data/imagenette_stage7_holdout')
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    excluded_files, excluded_pixels = prior_hashes([Path(args.pilot), Path(args.stage5)])
    selected, names, eligible, rejected = sample(
        args.validation_parquet, args.per_class, args.seed, excluded_files, excluded_pixels
    )
    fresh_files, fresh_pixels = set(), set()
    records = []
    for label, group in enumerate(selected):
        folder = out / 'images' / 'test' / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        for raw, digest, pixels, source_path in group:
            if digest in fresh_files or pixels in fresh_pixels:
                raise ValueError('Duplicate in newly selected holdout')
            fresh_files.add(digest)
            fresh_pixels.add(pixels)
            image_path = folder / (digest[:20] + '.jpg')
            image_path.write_bytes(raw)
            records.append({
                'image': image_path.relative_to(out).as_posix(), 'label': label,
                'question': 'What is the main object in this image?',
                'answer': names[label].lower(), 'sha256': digest,
                'pixel_sha256': pixels, 'source_path': source_path,
            })
    (out / 'test.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
        encoding='utf-8',
    )
    manifest = {
        'dataset': 'Imagenette 320px', 'purpose': 'new holdout for stage7 model',
        'source_parquet_sha256': sha256(args.validation_parquet),
        'source_dataset': 'https://huggingface.co/datasets/ShaomuTan/imagenette',
        'seed': args.seed, 'per_class': args.per_class,
        'count': len(records), 'class_names': names,
        'excluded_prior_file_hashes': len(excluded_files),
        'excluded_prior_pixel_hashes': len(excluded_pixels),
        'source_rows_rejected': rejected, 'source_rows_eligible_by_class': eligible,
        'prior_file_overlap': 0, 'prior_pixel_overlap': 0,
        'test_policy': 'Do not encode or evaluate until model configuration is frozen.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'count': len(records), 'rejected': rejected, 'eligible_by_class': eligible}), flush=True)


if __name__ == '__main__':
    main()
