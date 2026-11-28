"""Transmit author-VQGAN dev tokens through isolated Sionna demo PHY."""
import argparse
import json
from pathlib import Path

import numpy as np

from smoke_stage25_sionna_link import simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=Path, default=Path('runs/stage26_official_demo_tokens.npz'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--snr', type=float, default=8.)
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    with np.load(args.tokens) as source:
        tokens = source['tokens'].copy()
        source_meta = json.loads(source['metadata'].item())
    assert tokens.shape == (10, 256)
    received_all = np.empty((args.trials, 10, 256), np.int32)
    oracle_all = np.empty((args.trials, 10, 256), bool)
    crc_all = np.empty((args.trials, 10, 256), bool)
    rows = []
    for trial in range(args.trials):
        for image_index in range(10):
            seed = 20261112 + 100 * image_index + trial
            result, received, oracle_bad, crc_bad = simulate(
                seed, args.snr, tokens=tokens[image_index], return_arrays=True)
            assert np.array_equal(tokens[image_index][~np.tile(oracle_bad, 4)],
                                  received.reshape(-1)[~np.tile(oracle_bad, 4)])
            received_all[trial, image_index] = received.reshape(-1)
            oracle_all[trial, image_index] = np.tile(oracle_bad, 4)
            crc_all[trial, image_index] = np.tile(crc_bad, 4)
            rows.append({'trial': trial, 'image_index': image_index,
                         'source_index': source_meta['rows'][image_index]['source_index'],
                         'class_name': source_meta['rows'][image_index]['class_name'], **result})
        print('finished trial', trial, flush=True)
    keys = ('bit_errors', 'token_errors', 'oracle_bad_packets', 'crc_bad_packets',
            'crc_missed_bad_packets', 'crc_bad_but_data_correct_packets')
    metadata = {'scope': '10 stage5 dev images × independent trial seeds; author VQGAN tokens, demo-like Sionna PHY, no MaskGIT yet',
                'snr_db': args.snr, 'images': 10, 'trials': args.trials,
                'source_vqgan_sha256': source_meta['vqgan_sha256'],
                'source_indices': source_meta['selected_indices'],
                'totals': {k: sum(r[k] for r in rows) for k in keys}, 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, transmitted=tokens, received=received_all,
                        oracle_token_mask=oracle_all, crc_token_mask=crc_all,
                        metadata=json.dumps(metadata))
    print(json.dumps({'runs': len(rows), 'totals': metadata['totals']}), flush=True)


if __name__ == '__main__': main()
