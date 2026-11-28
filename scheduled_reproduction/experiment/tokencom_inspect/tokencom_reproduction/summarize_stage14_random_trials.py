"""Compare physical retransmission against five random draws per channel seed."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
cases = [json.loads((root / f'runs/stage14_random_trials_seed{s}/results.json')
                    .read_text(encoding='utf-8')) for s in range(20261029, 20261032)]
old_paths = [root / 'runs/stage13_physical_retransmission_val100/results.json'] + [
    root / f'runs/stage13_physical_retransmission_val100_seed{s}/results.json'
    for s in (20261030, 20261031)]
old = [json.loads(p.read_text(encoding='utf-8')) for p in old_paths]
assert all(len(c['rows']) == 100 and c['random_trials_per_image'] == 5 for c in cases)
assert [c['seed'] for c in cases] == [20261029, 20261030, 20261031]
assert all(a['data_identity'] == b['data_identity'] == cases[0]['data_identity']
           for a, b in zip(cases, old))
for case, previous in zip(cases, old):
    assert all(r['source_index'] == i and len(r['random_trials']) == 5 and
               r['low_confidence'] == previous['rows'][i]['low_confidence'] and
               r['random_trials'][0] == previous['rows'][i]['random']
               for i, r in enumerate(case['rows']))

seed_rows = []
for case in cases:
    rows = case['rows']
    missing = sum(r['initial_missing_tokens'] for r in rows)
    low = sum(r['low_confidence']['originally_missing_token_hits'] for r in rows)
    random_hits = [sum(r['random_trials'][t]['originally_missing_token_hits']
                       for r in rows) for t in range(5)]
    seed_rows.append({'seed': case['seed'], 'missing_tokens': missing,
                      'low_hits': low, 'random_trial_hits': random_hits,
                      'random_mean_hits': sum(random_hits) / 5,
                      'difference': (low - sum(random_hits) / 5) / missing})
by_image = []
for i in range(100):
    rows = [c['rows'][i] for c in cases]
    by_image.append({
        'source_index': i,
        'missing_tokens': sum(r['initial_missing_tokens'] for r in rows),
        'low_minus_random_mean_hits': sum(
            r['low_confidence']['originally_missing_token_hits'] -
            sum(t['originally_missing_token_hits'] for t in r['random_trials']) / 5
            for r in rows),
    })
missing = sum(r['missing_tokens'] for r in seed_rows)
low = sum(r['low_hits'] for r in seed_rows)
random = sum(r['random_mean_hits'] for r in seed_rows)
d = np.array([r['low_minus_random_mean_hits'] for r in by_image])
den = np.array([r['missing_tokens'] for r in by_image])
sample = np.random.default_rng(20261029).integers(0, 100, size=(10000, 100))
draws = d[sample].sum(1) / den[sample].sum(1)
result = {'scope': 'development physical link; three channel seeds, five random choices each',
          'unique_images': 100, 'missing_tokens_across_seeds': missing,
          'low_confidence_accuracy': low / missing,
          'five_trial_random_mean_accuracy': random / missing,
          'difference': (low - random) / missing,
          'image_cluster_bootstrap_95pct_difference':
              [float(x) for x in np.quantile(draws, [.025, .975])],
          'seed_rows': seed_rows,
          'caution': 'Same 100 images reused; random choices and channel draws are finite; no feedback overhead.'}
out = root / 'runs/stage14_random_trials_summary.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
