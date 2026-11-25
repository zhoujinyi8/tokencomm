"""Summarize paired validation-token accuracy with image-cluster bootstrap CIs.

These intervals describe this fixed validation experiment; they are not a blind
test, and pretrained ImageNet data overlap is not ruled out by this script.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path, draws, seed):
    source = json.loads(path.read_text(encoding='utf-8'))
    if source['test_accessed'] or source['split'] != 'stage5 validation only':
        raise ValueError('Expected validation-only result')
    if source['images'] != len(source['rows']):
        raise ValueError('Image count mismatch')
    denominators = np.array([row['missing_tokens'] for row in source['rows']], dtype=np.int64)
    if int(denominators.sum()) != source['missing_tokens']:
        raise ValueError('Missing-token count mismatch')
    methods = list(source['correct_counts'])
    counts = {method: np.array([row['correct'][method] for row in source['rows']], dtype=np.int64)
              for method in methods}
    for method in methods:
        if int(counts[method].sum()) != source['correct_counts'][method]:
            raise ValueError(f'Correct-count mismatch: {method}')
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, source['images'], size=(draws, source['images']))
    sampled_denominator = denominators[bootstrap_indices].sum(axis=1)
    spatial = counts['spatial']
    comparisons = {}
    for method in methods:
        if method == 'spatial': continue
        observed = (counts[method].sum() - spatial.sum()) / denominators.sum()
        sampled_difference = (counts[method][bootstrap_indices].sum(axis=1)
                              - spatial[bootstrap_indices].sum(axis=1)) / sampled_denominator
        comparisons[method] = {
            'difference_percentage_points': float(100 * observed),
            'image_bootstrap_95_percent_interval_pp':
                [float(100 * x) for x in np.quantile(sampled_difference, [.025, .975])],
        }
    return {
        'source': str(path), 'seed': source['seed'], 'per': source['per'],
        'images': source['images'], 'missing_tokens': source['missing_tokens'],
        'accuracy': source['accuracy'], 'comparisons_to_spatial': comparisons,
        'bootstrap_unit': 'image', 'bootstrap_draws': draws,
        'test_accessed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output file already exists')
    if args.draws < 1: parser.error('Draw count must be positive')
    analyses = [summarize(path, args.draws, args.seed + index)
                for index, path in enumerate(args.results)]
    result = {'scope': 'development validation only; paired image-cluster bootstrap',
              'test_accessed': False, 'analyses': analyses}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
