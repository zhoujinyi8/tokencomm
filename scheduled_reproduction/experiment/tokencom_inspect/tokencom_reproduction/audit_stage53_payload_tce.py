"""Supplemental analytic TCE check on complete frozen PHY caches, not model tuning."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from paper_tce_reference import ideal_arq_tce, payload_tce

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage53_payload_tce_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['script_sha256'] == sha(Path(__file__))
assert plan['reference_sha256'] == sha(root / 'paper_tce_reference.py')
assert plan['protocol_sha256'] == sha(run / 'stage44_frozen_protocol.json')
assert plan['original_primary_protocol_changed'] is False
assert plan['packet_error_definition'] == 'any received token ID differs from sent token in a fixed interleaved packet'
protocol = json.loads((run / 'stage44_frozen_protocol.json').read_text(encoding='utf-8'))
assert protocol['images'] == 100 and protocol['phy_trials'] == 5 and protocol['snr_db'] == [8.]
audit_path = run / 'stage45_cache_audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['complete'] and audit['encoded_images'] == 100 and audit['phy_cases'] == 500
expected_hashes = {name.replace('\\', '/'): digest for name, digest in audit['cache_sha256'].items()}
cases = []
for image in range(100):
    token_path = run / f'stage45_holdout_tokens/image{image:03d}.npz'
    assert sha(token_path) == expected_hashes[token_path.relative_to(root).as_posix()]
    with np.load(token_path) as tokens:
        tx = tokens['tokens'].copy()
        source = tokens['source_sha256'].item()
    for trial in range(5):
        path = run / f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz'
        assert sha(path) == expected_hashes[path.relative_to(root).as_posix()]
        with np.load(path) as cache:
            assert cache['source_sha256'].item() == source
            received = cache['received']
            oracle = cache['oracle_token_mask'].astype(bool).reshape(4, 64)
            crc = cache['crc_token_mask'].astype(bool).reshape(4, 64)
            errors = received != tx
            truth = errors.reshape(4, 64).any(axis=0)
            assert np.array_equal(oracle, np.broadcast_to(truth, (4, 64)))
            assert np.array_equal(crc, np.broadcast_to(crc[0], (4, 64)))
            flagged = crc[0]
            cases.append({'image_index': image, 'trial': trial, 'packets': 64,
                          'true_error_packets': int(truth.sum()),
                          'crc_failed_packets': int(flagged.sum()),
                          'data_correct_crc_failed_packets': int((~truth & flagged).sum()),
                          'data_wrong_crc_passed_packets': int((truth & ~flagged).sum()),
                          'wrong_received_tokens': int(errors.sum())})
totals = {key: sum(case[key] for case in cases) for key in cases[0] if key not in ('image_index', 'trial')}
assert len(cases) == 500 and totals['packets'] == 32000
per = totals['true_error_packets'] / totals['packets']
crc_failure_rate = totals['crc_failed_packets'] / totals['packets']
base = payload_tce()
arq = ideal_arq_tce(per)
assert math.isclose(arq, base * (1 - per), abs_tol=1e-12)
result = {'kind': 'supplemental descriptive analytic calculation; not a preregistered primary outcome',
          'created_utc': datetime.now(timezone.utc).isoformat(),
          'source': plan['source'], 'plan_sha256': sha(plan_path),
          'script_sha256': sha(Path(__file__)), 'reference_sha256': plan['reference_sha256'],
          'protocol_sha256': plan['protocol_sha256'], 'cache_audit_sha256': sha(audit_path),
          'complete_images': 100, 'complete_receptions': 500,
          'input_hashes_matched': True, 'original_primary_protocol_changed': False,
          'image_quality_metrics_read': False, 'strategy_or_threshold_selected': False,
          'totals': totals, 'true_packet_error_rate': per, 'crc_failure_rate': crc_failure_rate,
          'units': 'pixels/token_payload_bit', 'single_shot_tce': base,
          'ideal_arq_mean_attempts_from_pooled_per': 1/(1-per) if per < 1 else None,
          'ideal_arq_tce_from_true_per': arq,
          'assumptions': plan['assumptions'], 'cases': cases}
out = run / 'stage53_payload_tce_result.json'
assert not out.exists(), 'Preserve an existing analytic result'
out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
