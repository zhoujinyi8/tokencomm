"""Aggregate three physical-link development seeds by image cluster."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
paths = [ROOT / 'runs/stage13_physical_retransmission_val100/results.json'] + [
    ROOT / f'runs/stage13_physical_retransmission_val100_seed{s}/results.json'
    for s in (20261030, 20261031)]
cases = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
assert [case['seed'] for case in cases] == [20261029, 20261030, 20261031]
assert all(len(case['rows']) == 100 and case['budget_packets'] == 4 and
           case['channel'] == 'rayleigh' and case['snr_db'] == 8. for case in cases)
assert len({case['checkpoint_sha256'] for case in cases}) == 1
assert len({json.dumps(case['data_identity'], sort_keys=True) for case in cases}) == 1
for case in cases:
    assert [row['source_index'] for row in case['rows']] == list(range(100))

seed_rows = []
for case in cases:
    rows = case['rows']
    n = sum(r['initial_missing_tokens'] for r in rows)
    low = sum(r['low_confidence']['originally_missing_token_hits'] for r in rows)
    random = sum(r['random']['originally_missing_token_hits'] for r in rows)
    seed_rows.append({
        'seed': case['seed'], 'initial_missing_tokens': n,
        'initial_hits': sum(r['initial_originally_missing_token_hits'] for r in rows),
        'low_confidence_hits': low, 'random_hits': random,
        'low_confidence_retransmission_crc_passes': sum(
            r['low_confidence']['crc_passed_retransmissions'] for r in rows),
        'random_retransmission_crc_passes': sum(
            r['random']['crc_passed_retransmissions'] for r in rows),
        'difference': (low - random) / n,
    })
by_image = []
for index in range(100):
    rows = [case['rows'][index] for case in cases]
    by_image.append({
        'source_index': index,
        'initial_missing_tokens': sum(r['initial_missing_tokens'] for r in rows),
        'low_minus_random_hits': sum(
            r['low_confidence']['originally_missing_token_hits'] -
            r['random']['originally_missing_token_hits'] for r in rows),
    })
n = sum(r['initial_missing_tokens'] for r in seed_rows)
low = sum(r['low_confidence_hits'] for r in seed_rows)
random = sum(r['random_hits'] for r in seed_rows)
d = np.array([r['low_minus_random_hits'] for r in by_image])
den = np.array([r['initial_missing_tokens'] for r in by_image])
sample = np.random.default_rng(20261029).integers(0, 100, size=(10000, 100))
draws = d[sample].sum(1) / den[sample].sum(1)
result = {
    'scope': 'exploratory development physical retransmission; three seeds reuse 100 images',
    'seeds': seed_rows, 'unique_images': 100,
    'missing_tokens_across_seeds': n,
    'low_confidence_accuracy': low / n,
    'single_draw_random_accuracy': random / n,
    'difference': (low - random) / n,
    'image_cluster_bootstrap_95pct_difference':
        [float(x) for x in np.quantile(draws, [.025, .975])],
    'caution': 'Random comparator is one draw per seed; feedback overhead is excluded.',
}
out = ROOT / 'runs/stage13_physical_retransmission_summary.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
