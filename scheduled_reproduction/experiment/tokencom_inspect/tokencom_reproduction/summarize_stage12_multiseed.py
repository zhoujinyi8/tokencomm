"""Summarize five channel seeds with images as bootstrap clusters."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PATHS = [ROOT / 'runs/stage12_retransmission_priority_val/results.json'] + [
    ROOT / f'runs/stage12_retransmission_priority_val_seed{seed}/results.json'
    for seed in range(20261022, 20261026)]
cases = [json.loads(path.read_text(encoding='utf-8')) for path in PATHS]
assert [case['seed'] for case in cases] == list(range(20261021, 20261026))
assert all(case['per'] == .3 and case['budget_packets_per_image'] == 4 and
           case['images'] == 100 for case in cases)
assert len({json.dumps(case['data_identity'], sort_keys=True) for case in cases}) == 1
assert len({case['checkpoint_sha256'] for case in cases}) == 1
for case in cases:
    assert [r['source_index'] for r in case['rows']] == list(range(100))
    assert sum(r['missing_tokens'] for r in case['rows']) == case['missing_tokens']

per_image = []
for index in range(100):
    rows = [case['rows'][index] for case in cases]
    assert len({r['class'] for r in rows}) == 1
    per_image.append({
        'source_index': index, 'class': rows[0]['class'],
        'missing_tokens': sum(r['missing_tokens'] for r in rows),
        'initial_correct': sum(r['baseline_correct'] for r in rows),
        'uncertain_corrected_errors': sum(r['uncertain_corrected_errors'] for r in rows),
        'random_expected_corrected_errors': sum(
            r['random_expected_corrected_errors'] for r in rows),
    })
missing = sum(r['missing_tokens'] for r in per_image)
base = sum(r['initial_correct'] for r in per_image)
uncertain = sum(r['uncertain_corrected_errors'] for r in per_image)
random = sum(r['random_expected_corrected_errors'] for r in per_image)
delta = np.array([r['uncertain_corrected_errors'] -
                  r['random_expected_corrected_errors'] for r in per_image])
denom = np.array([r['missing_tokens'] for r in per_image])
rng = np.random.default_rng(20261028)
samples = rng.integers(0, 100, size=(10000, 100))
draws = delta[samples].sum(axis=1) / denom[samples].sum(axis=1)
result = {
    'scope': 'exploratory stage5 validation; five channel seeds reuse same 100 images',
    'seeds': [case['seed'] for case in cases], 'per': .3,
    'budget_packets_per_image': 4, 'unique_images': 100,
    'image_seed_cases': 500, 'missing_tokens_across_seeds': missing,
    'initial_maskgit_accuracy': base / missing,
    'post_retransmit_accuracy': {
        'low_confidence': (base + uncertain) / missing,
        'random_expectation': (base + random) / missing},
    'difference': (uncertain - random) / missing,
    'image_cluster_bootstrap_95pct_difference':
        [float(x) for x in np.quantile(draws, [.025, .975])],
    'seed_differences': [case['low_confidence_minus_random_expected'] for case in cases],
    'repeated_image_caution': 'Bootstrap resamples 100 images, keeping all five seed outcomes together.',
    'rows': per_image,
}
out = ROOT / 'runs/stage12_retransmission_priority_multiseed_summary.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ['missing_tokens_across_seeds',
      'post_retransmit_accuracy', 'difference',
      'image_cluster_bootstrap_95pct_difference', 'seed_differences']}, indent=2))
