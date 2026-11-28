"""Repeat the isolated old-Sionna PHY check across independent seeds."""
import argparse
import json
from pathlib import Path

from smoke_stage25_sionna_link import simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trials', type=int, default=20)
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    groups = {}
    for snr in (4., 8., 12., 40.):
        rows = [simulate(20261112 + i, snr) for i in range(args.trials)]
        assert all(r['raw_bits_per_packet'] == 40 and r['crc_bits_per_packet'] == 51
                   and r['coded_bits_per_packet'] == 102 and r['qam_symbols_total'] == 1632 for r in rows)
        keys = ('bit_errors', 'token_errors', 'oracle_bad_packets', 'crc_bad_packets',
                'crc_missed_bad_packets', 'crc_bad_but_data_correct_packets')
        groups[str(snr)] = {'trials': len(rows), 'packets': len(rows) * 64,
                            'totals': {key: sum(r[key] for r in rows) for key in keys},
                            'rows': rows}
        print(snr, groups[str(snr)]['totals'], flush=True)
    assert groups['40.0']['totals']['bit_errors'] == 0
    result = {'scope': 'synthetic tokens, independent noise and token seeds; no image model',
              'parameters': 'Sionna 0.19.2, TensorFlow 2.15.1, CRC11, conv 101/111, 16-QAM, AWGN',
              'seed_start': 20261112, 'groups': groups}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
