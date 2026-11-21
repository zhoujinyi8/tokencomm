"""Verify header-bearing live feedback against cached physical data outcomes."""
import json
from pathlib import Path

import numpy as np

from digital_link import bits_int, int_bits, transmit_packet

root = Path(__file__).resolve().parent
live = json.loads((root / 'runs/stage19_header16_live_verify10/results.json').read_text(encoding='utf-8'))
cached = json.loads((root / 'runs/stage14_random_trials_seed20261029/results.json').read_text(encoding='utf-8'))
assert len(live['rows']) == 10 and live['random_trials_per_image'] == 5
assert live['feedback_header_bits'] == 16 and live['feedback_attempts'] == 2
assert live['seed'] == cached['seed'] == live['feedback_seed'] == 20261029
assert live['checkpoint_sha256'] == cached['checkpoint_sha256']
checks = 0
failures = 0
for actual, saved in zip(live['rows'], cached['rows']):
    index = actual['source_index']
    assert index == saved['source_index']
    assert actual['initial_missing_tokens'] == saved['initial_missing_tokens']
    assert actual['initial_originally_missing_token_hits'] == saved['initial_originally_missing_token_hits']
    for got, expected in [(actual['low_confidence'], saved['low_confidence'])] + list(
            zip(actual['random_trials'], saved['random_trials'])):
        assert got['selected_packet_ids'] == expected['selected_packet_ids']
        header = int_bits((1 << 12) | index, 16).reshape(-1)
        bits = np.concatenate((header, int_bits(np.asarray(got['selected_packet_ids']), 6).reshape(-1)))
        rng = np.random.default_rng(live['feedback_seed'] + live['seed'] * 1000 + index)
        ok = False
        for attempt in (1, 2):
            decoded, ok, detail = transmit_packet(bits, live['snr_db'], rng, 'rayleigh')
            assert detail['symbols'] == 31 and detail['coded_bits'] == 124
            if ok:
                assert np.array_equal(decoded[:16], header)
                assert bits_int(decoded[16:].reshape(4, 6)).tolist() == got['decoded_feedback_packet_ids']
                break
        assert got['feedback_crc_passed'] == ok
        assert got['feedback_attempts_used'] == attempt
        if ok:
            assert got['decoded_feedback_packet_ids'] == got['selected_packet_ids']
            assert got['originally_missing_token_hits'] == expected['originally_missing_token_hits']
            assert got['remaining_missing_tokens'] == expected['remaining_missing_tokens']
            assert got['crc_passed_retransmissions'] == expected['crc_passed_retransmissions']
        else:
            failures += 1
            assert got['decoded_feedback_packet_ids'] == []
            assert got['originally_missing_token_hits'] == actual['initial_originally_missing_token_hits']
            assert got['remaining_missing_tokens'] == actual['initial_missing_tokens']
            assert got['crc_passed_retransmissions'] == 0
        checks += 1
print(f'Header live replay verified: {len(live["rows"])} images, {checks} requests, '
      f'{failures} double-feedback failures')
