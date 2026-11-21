"""Isolated old-Sionna check of the public TokCom demo's 64-packet PHY chain.

Uses the same Sionna components/parameters as the demo. Noise RNG is TensorFlow,
so sample-by-sample outputs are not expected to match the torch notebook RNG.
"""
import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf
# Sionna 0.19 imports optional ray tracing at package import time. Its Windows
# CPU build requires LLVM-C.dll, although none of the PHY modules below use RT.
sys.modules['sionna.rt'] = types.ModuleType('sionna.rt')
from sionna.fec.crc import CRCEncoder, CRCDecoder
from sionna.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.mapping import Mapper, Demapper


def simulate(seed, snr, tokens=None, return_arrays=False, return_soft=False):
    rng = np.random.default_rng(seed)
    if tokens is None:
        tokens = rng.integers(0, 1024, size=(1, 256), dtype=np.int32)
    else:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(1, 256)
        if np.any((tokens < 0) | (tokens >= 1024)):
            raise ValueError('Expected 256 ten-bit VQGAN token IDs')
    packages = tokens.reshape(1, 4, 64).transpose(0, 2, 1)
    bits = ((packages[..., None] >> np.arange(9, -1, -1)) & 1).astype(np.float32)
    bits = bits.reshape(1, 64, 40)
    crc_encoder = CRCEncoder('CRC11')
    crc_decoder = CRCDecoder(crc_encoder)
    conv_encoder = ConvEncoder(gen_poly=['101', '111'])
    conv_decoder = ViterbiDecoder(encoder=conv_encoder)
    mapper = Mapper('qam', num_bits_per_symbol=4)
    demapper = Demapper(demapping_method='app', constellation_type='qam', num_bits_per_symbol=4)
    tx = tf.convert_to_tensor(bits)
    crc = crc_encoder(tx)
    coded = conv_encoder(crc)
    symbols = mapper(tf.reshape(coded, [1, -1, 4]))
    no = 10. ** (-snr / 10.)
    tf.random.set_seed(seed)
    noise = tf.complex(tf.random.normal(tf.shape(symbols), stddev=math.sqrt(no/2)),
                       tf.random.normal(tf.shape(symbols), stddev=math.sqrt(no/2)))
    llr = demapper((symbols + noise, no))
    decoded_crc = conv_decoder(tf.reshape(llr, [1, 64, 102]))
    decoded, crc_valid = crc_decoder(decoded_crc)
    rx_bits = np.rint(decoded.numpy()).astype(np.int32).reshape(1, 64, 4, 10)
    rx_packages = np.sum(rx_bits * (1 << np.arange(9, -1, -1)), axis=-1)
    received = rx_packages.transpose(0, 2, 1).reshape(1, 256)
    oracle_bad = (packages != rx_packages).any(axis=-1).reshape(-1)
    crc_bad = ~crc_valid.numpy().reshape(-1).astype(bool)
    result = {
        'snr_db': snr, 'seed': seed,
        'raw_bits_per_packet': int(bits.shape[-1]),
        'crc_bits_per_packet': int(crc.shape[-1]),
        'coded_bits_per_packet': int(coded.shape[-1]),
        'qam_symbols_total': int(np.prod(symbols.shape[1:])),
        'bit_errors': int(np.count_nonzero(bits != decoded.numpy())),
        'token_errors': int(np.count_nonzero(tokens != received)),
        'oracle_bad_packets': int(oracle_bad.sum()),
        'crc_bad_packets': int(crc_bad.sum()),
        'crc_missed_bad_packets': int((oracle_bad & ~crc_bad).sum()),
        'crc_bad_but_data_correct_packets': int((~oracle_bad & crc_bad).sum()),
    }
    if return_soft:
        details = {'coded_llr': llr.numpy().reshape(64, 102),
                   'decoded_crc_bits': decoded_crc.numpy().reshape(64, 51),
                   'reencoded_decoded_bits': conv_encoder(decoded_crc).numpy().reshape(64, 102)}
        return result, received, oracle_bad, crc_bad, details
    return (result, received, oracle_bad, crc_bad) if return_arrays else result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output exists')
    rows = [simulate(20261111, snr) for snr in (40., 8., 4.)]
    assert rows[0]['bit_errors'] == rows[0]['token_errors'] == rows[0]['oracle_bad_packets'] == 0
    assert all(r['raw_bits_per_packet'] == 40 and r['crc_bits_per_packet'] == 51
               and r['coded_bits_per_packet'] == 102 and r['qam_symbols_total'] == 1632 for r in rows)
    result = {'scope': 'synthetic tokens; isolated Sionna physical link, not image reconstruction',
              'interleaver': 'fixed token index mod 64, four tokens per packet',
              'rows': rows, 'tensorflow_version': tf.__version__,
              'sionna_version': '0.19.2',
              'note': 'CRC can fail when parity bits change even if decoded 40 data bits and tokens match; this is not a CRC false alarm.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
