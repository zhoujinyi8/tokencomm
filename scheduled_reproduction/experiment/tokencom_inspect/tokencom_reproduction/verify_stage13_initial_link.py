"""Verify stage13's explicit packet loop matches the shared physical link."""
import numpy as np
import torch

from digital_link import bits_int, int_bits, transmit_packet, transmit_tokens

cache = torch.load('data/imagenette_stage5/token_cache/val.pt',
                   map_location='cpu', weights_only=True)
ids = cache['ids'][:1].numpy().astype(np.int64)
for seed in (20261029, 20261030, 20261031):
    expected_rec, expected_missing, expected_stats = transmit_tokens(
        ids, snr_db=8., channel='rayleigh', seed=seed, packet_tokens=4)
    rng = np.random.default_rng(seed)
    order = rng.permutation(256)
    rec = np.zeros_like(ids)
    missing = np.zeros_like(ids, dtype=bool)
    for packet in range(64):
        pos = order[packet * 4:(packet + 1) * 4]
        payload = int_bits(ids[0, pos], 10).reshape(-1)
        decoded, ok, _ = transmit_packet(payload, 8., rng, 'rayleigh')
        if ok: rec[0, pos] = bits_int(decoded.reshape(-1, 10))
        else: missing[0, pos] = True
    assert np.array_equal(rec, expected_rec), seed
    assert np.array_equal(missing, expected_missing), seed
    assert int(missing.sum()) == 4 * expected_stats['crc_failed_packets'], seed
print('PASS: stage13 initial receiver tokens and CRC erasures match transmit_tokens for all three seeds')
