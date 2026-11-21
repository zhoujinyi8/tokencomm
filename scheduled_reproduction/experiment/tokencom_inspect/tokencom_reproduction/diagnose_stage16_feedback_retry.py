"""Development-only feedback retry over cached physical data retransmissions.

Receiver sends four packet IDs. On a CRC failure it retries once after an
assumed timeout. A CRC-passing decoded ID list must match the cached request;
otherwise the cached data outcome cannot be used and the script stops.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from digital_link import bits_int, int_bits, transmit_packet


def run():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feedback-seed', type=int, default=20261029)
    p.add_argument('--max-attempts', type=int, default=2)
    p.add_argument('--header-bits', type=int, choices=(0, 16), default=0,
                   help='16-bit control header: 4-bit type plus 12-bit image sequence')
    p.add_argument('--feedback-snr', type=float, default=8.)
    p.add_argument('--out', type=Path,
                   default=root / 'runs/stage16_feedback_retry_summary.json')
    args = p.parse_args()
    if args.max_attempts < 1:
        p.error('max-attempts must be positive')
    if args.out.exists():
        p.error('Output already exists')
    cases = [json.loads((root / f'runs/stage14_random_trials_seed{s}/results.json')
                        .read_text(encoding='utf-8')) for s in range(20261029, 20261032)]
    assert all(c['seed'] == s and len(c['rows']) == 100 and
               c['random_trials_per_image'] == 5 and c['snr_db'] == 8. and
               c['channel'] == 'rayleigh' for s, c in zip(range(20261029, 20261032), cases))
    totals = {name: {'hits': 0., 'feedback_attempts': 0,
                     'feedback_crc_passes': 0, 'data_retransmissions': 0,
                     'request_count': 0, 'attempt_histogram': {str(i): 0 for i in range(1, args.max_attempts + 1)}}
              for name in ['low', 'random']}
    by_image = {i: {'missing': 0, 'low': 0., 'random': 0.} for i in range(100)}

    symbols_per_feedback = 23 if args.header_bits == 0 else 31

    def request(selected, after_hits, before_hits, state, index):
        id_bits = int_bits(np.asarray(selected, dtype=np.int64), 6).reshape(-1)
        header = int_bits((1 << 12) | index, 16).reshape(-1) if args.header_bits else np.empty(0, dtype=np.uint8)
        bits = np.concatenate((header, id_bits))
        assert len(id_bits) == 24 and len(set(selected)) == 4 and all(0 <= x < 64 for x in selected)
        local = np.random.default_rng()
        local.bit_generator.state = copy.deepcopy(state)
        for attempt in range(1, args.max_attempts + 1):
            decoded, ok, detail = transmit_packet(bits, args.feedback_snr, local, 'rayleigh')
            assert detail['symbols'] == symbols_per_feedback and detail['coded_bits'] == 2 * (len(bits) + 16 + 6)
            if ok:
                if args.header_bits and not np.array_equal(decoded[:16], header):
                    raise RuntimeError('CRC-undetected control header error; cached retransmission is invalid')
                decoded_ids = bits_int(decoded[args.header_bits:].reshape(4, 6)).tolist()
                if decoded_ids != selected:
                    raise RuntimeError(f'CRC-undetected control error: decoded {decoded_ids}, expected {selected}; '
                                       'cached data retransmission is invalid')
                return after_hits, attempt, True
        return before_hits, args.max_attempts, False

    for case in cases:
        for row in case['rows']:
            index = row['source_index']
            rng = np.random.default_rng(args.feedback_seed + case['seed'] * 1000 + index)
            state = copy.deepcopy(rng.bit_generator.state)
            before = row['initial_originally_missing_token_hits']
            missing = row['initial_missing_tokens']
            by_image[index]['missing'] += missing
            for name, trials in [('low', [row['low_confidence']]),
                                 ('random', row['random_trials'])]:
                hits = []
                for trial in trials:
                    correct, attempts, passed = request(
                        trial['selected_packet_ids'], trial['originally_missing_token_hits'],
                        before, state, index)
                    hits.append(correct)
                    stat = totals[name]
                    stat['request_count'] += 1
                    stat['hits'] += correct / len(trials)
                    stat['feedback_attempts'] += attempts
                    stat['feedback_crc_passes'] += int(passed)
                    stat['data_retransmissions'] += 4 * int(passed)
                    stat['attempt_histogram'][str(attempts)] += 1
                by_image[index][name] += sum(hits) / len(hits)

    missing = sum(x['missing'] for x in by_image.values())
    delta = np.array([by_image[i]['low'] - by_image[i]['random'] for i in range(100)])
    denominator = np.array([by_image[i]['missing'] for i in range(100)])
    draw = np.random.default_rng(20261029).integers(0, 100, size=(10000, 100))
    interval = np.quantile(delta[draw].sum(1) / denominator[draw].sum(1), [.025, .975])
    for stat in totals.values():
        stat['accuracy'] = stat['hits'] / missing
        stat['feedback_symbols_total'] = symbols_per_feedback * stat['feedback_attempts']
        stat['data_retransmission_symbols_total'] = 31 * stat['data_retransmissions']
        stat['added_symbols_per_request'] = (
            stat['feedback_symbols_total'] + stat['data_retransmission_symbols_total']) / stat['request_count']
        stat['added_symbols_pct_of_initial_1984'] = stat['added_symbols_per_request'] / 1984
    result = {
        'scope': 'development-only feedback retry with decoded packet IDs and cached physical data outcomes',
        'feedback_seed': args.feedback_seed, 'max_feedback_attempts': args.max_attempts,
        'feedback_snr_db': args.feedback_snr,
        'header_bits': args.header_bits, 'feedback_symbols_per_attempt': symbols_per_feedback,
        'images': 100, 'data_channel_seeds': [c['seed'] for c in cases],
        'missing_tokens_across_seeds': missing,
        'low': totals['low'], 'random_five_trial_mean': totals['random'],
        'accuracy_difference': (totals['low']['hits'] - totals['random']['hits']) / missing,
        'image_cluster_bootstrap_95pct_difference': [float(x) for x in interval],
        'limitations': 'Feedback retry assumes timeout without a modeled duration; no ACK, pilot or control-error recovery. '
                       'Cached data results apply only to exactly decoded packet IDs.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
