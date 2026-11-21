"""Check live decoded-feedback retransmission against cached-data replay."""
import json
from pathlib import Path

import numpy as np
from digital_link import bits_int, int_bits, transmit_packet

root = Path(__file__).resolve().parent
cached = json.loads((root / 'runs/stage14_random_trials_seed20261029/results.json').read_text(encoding='utf-8'))
for n, trials in [(10, 5), (100, 1)]:
    path = root / f'runs/stage16_live_feedback_verify{n}/results.json'
    assert path.exists(), f'Missing live run: {path}'
    live = json.loads(path.read_text(encoding='utf-8'))
    assert live['seed'] == cached['seed'] == 20261029
    assert live['feedback_attempts'] == 2 and live['feedback_seed'] == 20261029
    assert live['checkpoint_sha256'] == cached['checkpoint_sha256']
    assert len(live['rows']) == n and live['random_trials_per_image'] == trials
    checks = 0
    failed_feedback = 0
    for actual, saved in zip(live['rows'], cached['rows']):
        assert actual['source_index'] == saved['source_index']
        assert actual['initial_missing_tokens'] == saved['initial_missing_tokens']
        assert actual['initial_originally_missing_token_hits'] == saved['initial_originally_missing_token_hits']
        for got, expected in [(actual['low_confidence'], saved['low_confidence'])] + list(
                zip(actual['random_trials'], saved['random_trials'])):
            assert got['selected_packet_ids'] == expected['selected_packet_ids']
            control_rng = np.random.default_rng(live['feedback_seed'] + live['seed'] * 1000 + actual['source_index'])
            bits = int_bits(np.asarray(got['selected_packet_ids'], dtype=np.int64), 6).reshape(-1)
            expected_ok = False
            for attempt in range(1, live['feedback_attempts'] + 1):
                decoded, ok, _ = transmit_packet(bits, live['snr_db'], control_rng, 'rayleigh')
                if ok:
                    expected_ok = True
                    assert bits_int(decoded.reshape(4, 6)).tolist() == got['decoded_feedback_packet_ids']
                    break
            assert got['feedback_crc_passed'] == expected_ok
            assert got['feedback_attempts_used'] == attempt
            if got['feedback_crc_passed']:
                assert got['decoded_feedback_packet_ids'] == got['selected_packet_ids']
                assert got['originally_missing_token_hits'] == expected['originally_missing_token_hits']
                assert got['remaining_missing_tokens'] == expected['remaining_missing_tokens']
                assert got['crc_passed_retransmissions'] == expected['crc_passed_retransmissions']
            else:
                failed_feedback += 1
                assert got['decoded_feedback_packet_ids'] == []
                assert got['originally_missing_token_hits'] == actual['initial_originally_missing_token_hits']
                assert got['remaining_missing_tokens'] == actual['initial_missing_tokens']
                assert got['crc_passed_retransmissions'] == 0
            checks += 1
    print(f'Live feedback replay verified: {n} images, {checks} requests, '
          f'{failed_feedback} requests with both feedback attempts failing')
