"""Payload-only TCE reference for arXiv:2502.12096v4, section VI.

This module reads no experimental data. It does not change frozen evaluation.
CRC, channel coding, modulation, CMI, feedback and latency costs are excluded.
"""
import json
import math

SOURCE = 'https://arxiv.org/html/2502.12096v4#S6'


def payload_tce(height=256, width=256, token_count=256, codebook_size=1024, mean_attempts=1.0):
    """Return pixels per token payload bit, counting the first attempt in T."""
    for value in (height, width, token_count, codebook_size):
        if type(value) is not int or value <= 0:
            raise ValueError('Dimensions, token count and codebook size must be positive integers')
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError('This reference supports power-of-two token codebooks')
    if isinstance(mean_attempts, bool) or not isinstance(mean_attempts, (int, float)):
        raise ValueError('Mean attempts must be numeric and at least one')
    if math.isnan(mean_attempts) or mean_attempts < 1:
        raise ValueError('Mean attempts must be at least one')
    return height * width / (mean_attempts * token_count * math.log2(codebook_size))


def ideal_arq_tce(packet_error_rate, **tokenizer):
    """Apply the ideal geometric-attempt model; PER=1 gives zero efficiency."""
    if isinstance(packet_error_rate, bool) or not isinstance(packet_error_rate, (int, float)):
        raise ValueError('Packet error rate must be numeric')
    if not math.isfinite(packet_error_rate) or not 0 <= packet_error_rate <= 1:
        raise ValueError('Packet error rate must be in [0,1]')
    if 'mean_attempts' in tokenizer:
        raise ValueError('The ideal ARQ model determines mean attempts')
    return payload_tce(**tokenizer) * (1 - packet_error_rate)


if __name__ == '__main__':
    rows = [{'packet_error_rate': p, 'single_shot_tce': payload_tce(),
             'ideal_arq_tce': ideal_arq_tce(p)} for p in (0.0, 0.2, 0.4, 1.0)]
    print(json.dumps({'source': SOURCE, 'units': 'pixels/token_payload_bit',
                      'experimental_data_accessed': False, 'analytic_examples': rows},
                     indent=2, allow_nan=False))
