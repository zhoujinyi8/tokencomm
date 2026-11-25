"""Development-only coded feedback gate applied to stored stage14 outcomes.

Each request carries four six-bit packet IDs in one CRC-protected Rayleigh
control packet. If feedback CRC fails, no data retransmission occurs. This
reuses cached data-retransmission outcomes only when feedback decoded exactly.
"""
import copy
import argparse
import json
from pathlib import Path

import numpy as np

from digital_link import int_bits, transmit_packet

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--feedback-seed', type=int, default=20261029)
p.add_argument('--out', type=Path,
               default=root / 'runs/stage15_feedback_gate_summary.json')
args = p.parse_args()
cases = [json.loads((root / f'runs/stage14_random_trials_seed{s}/results.json')
                    .read_text(encoding='utf-8')) for s in range(20261029, 20261032)]
assert all(len(c['rows']) == 100 and c['random_trials_per_image'] == 5 and
           c['snr_db'] == 8. and c['channel'] == 'rayleigh' for c in cases)
stats = []
by_image = {i: {'missing': 0, 'low': 0, 'random_mean': 0.} for i in range(100)}
for case in cases:
    totals = {'missing_tokens': 0, 'low_hits': 0, 'random_mean_hits': 0.,
              'low_feedback_crc_passes': 0, 'random_feedback_crc_passes': 0,
              'random_feedback_packets': 0}
    for row in case['rows']:
        index = row['source_index']
        rng = np.random.default_rng(args.feedback_seed + case['seed'] * 1000 + index)
        state = copy.deepcopy(rng.bit_generator.state)

        def gated(selected, hit_after):
            bits = int_bits(np.asarray(selected, dtype=np.int64), 6).reshape(-1)
            local = np.random.default_rng()
            local.bit_generator.state = copy.deepcopy(state)
            decoded, ok, _ = transmit_packet(bits, 8., local, 'rayleigh')
            if ok and not np.array_equal(decoded, bits):
                raise ValueError('Undetected feedback bit error; cached retransmission cannot apply')
            return (hit_after if ok else row['initial_originally_missing_token_hits']), int(ok)

        low = row['low_confidence']
        low_hits, low_ok = gated(low['selected_packet_ids'],
                                 low['originally_missing_token_hits'])
        random_hits = []
        for trial in row['random_trials']:
            hits, ok = gated(trial['selected_packet_ids'],
                             trial['originally_missing_token_hits'])
            random_hits.append(hits)
            totals['random_feedback_crc_passes'] += ok
            totals['random_feedback_packets'] += 1
        totals['missing_tokens'] += row['initial_missing_tokens']
        totals['low_hits'] += low_hits
        totals['random_mean_hits'] += sum(random_hits) / len(random_hits)
        totals['low_feedback_crc_passes'] += low_ok
        by_image[index]['missing'] += row['initial_missing_tokens']
        by_image[index]['low'] += low_hits
        by_image[index]['random_mean'] += sum(random_hits) / len(random_hits)
    stats.append({'seed': case['seed'], **totals})

missing = sum(s['missing_tokens'] for s in stats)
low = sum(s['low_hits'] for s in stats)
random = sum(s['random_mean_hits'] for s in stats)
delta = np.array([by_image[i]['low'] - by_image[i]['random_mean'] for i in range(100)])
denom = np.array([by_image[i]['missing'] for i in range(100)])
draw = np.random.default_rng(20261029).integers(0, 100, size=(10000, 100))
interval = np.quantile(delta[draw].sum(1) / denom[draw].sum(1), [.025, .975])
result = {
    'scope': 'development-only coded feedback gate; cached data retransmissions',
    'feedback_seed': args.feedback_seed,
    'images': 100, 'channel_seeds': [s['seed'] for s in stats],
    'feedback_channel': 'independent Rayleigh Es/N0=8 dB, 24 payload bits, CRC+convolutional code+16-QAM',
    'feedback_coded_bits_per_image': 92, 'feedback_symbols_per_image': 23,
    'missing_tokens_across_seeds': missing,
    'low_accuracy': low / missing, 'random_five_trial_mean_accuracy': random / missing,
    'difference': (low - random) / missing,
    'image_cluster_bootstrap_95pct_difference': [float(x) for x in interval],
    'seed_stats': stats,
    'limitation': 'No feedback packet header, ACK, latency, or retry; only one feedback attempt.'}
out = args.out
if out.exists() and args.feedback_seed != 20261029:
    raise ValueError('Output exists')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
