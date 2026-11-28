"""Describe full-pilot CRC/payload disagreement and masking exposure, not model quality."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage54_development_multisnr_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
audit_path = run / 'stage54_development_phy_audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['all_checks_passed'] and audit['complete'] and audit['completed_cases'] == 300
assert audit['plan_sha256'] == sha(plan_path)
assert len(audit['complete_snr_groups']) == 6
source = run / 'stage26_official_token_link_snr8_5trials.npz'
assert sha(source) == plan['input_sha256']['runs/stage26_official_token_link_snr8_5trials.npz']
with np.load(source) as saved:
    sent = saved['transmitted'].copy()
groups = {}
for si, snr in enumerate(plan['snr_db']):
    counts = dict(true_error_packets=0, crc_failed_packets=0, crc_failed_data_correct_packets=0,
                  crc_passed_data_wrong_packets=0, wrong_tokens=0,
                  oracle_masked_tokens=0, crc_masked_tokens=0,
                  oracle_masked_originally_correct_tokens=0, crc_masked_originally_correct_tokens=0)
    for image in range(10):
        for trial in range(5):
            name = f'snr{si}_image{image:02d}_trial{trial}.npz'
            path = run / 'stage54_development_phy' / name
            assert sha(path) == audit['case_sha256'][name]
            with np.load(path) as saved:
                assert saved['plan_sha256'].item() == sha(plan_path)
                wrong = saved['received'] != sent[image]
                bad = wrong.reshape(4, 64).any(axis=0)
                oracle = saved['oracle_token_mask'].astype(bool)
                crc = saved['crc_token_mask'].astype(bool)
                crc_bad = crc.reshape(4, 64)[0]
                assert np.array_equal(oracle.reshape(4, 64), np.broadcast_to(bad, (4, 64)))
                assert np.array_equal(crc.reshape(4, 64), np.broadcast_to(crc_bad, (4, 64)))
                values = {'true_error_packets': bad.sum(), 'crc_failed_packets': crc_bad.sum(),
                          'crc_failed_data_correct_packets': (crc_bad & ~bad).sum(),
                          'crc_passed_data_wrong_packets': (~crc_bad & bad).sum(), 'wrong_tokens': wrong.sum(),
                          'oracle_masked_tokens': oracle.sum(), 'crc_masked_tokens': crc.sum(),
                          'oracle_masked_originally_correct_tokens': (oracle & ~wrong).sum(),
                          'crc_masked_originally_correct_tokens': (crc & ~wrong).sum()}
                for key, value in values.items():
                    counts[key] += int(value)
    reference = audit['complete_snr_groups'][str(snr)]
    assert counts['true_error_packets']/3200 == reference['true_packet_error_rate']
    assert counts['crc_failed_packets']/3200 == reference['crc_failure_rate']
    assert counts['wrong_tokens']/12800 == reference['received_token_error_rate']
    assert counts['crc_failed_packets'] == counts['true_error_packets']-counts['crc_passed_data_wrong_packets']+counts['crc_failed_data_correct_packets']
    groups[str(snr)] = dict(complete_receptions=50, images=10, packets=3200, tokens=12800, **counts)
result = {'scope': plan['scope'], 'plan_sha256': sha(plan_path), 'phy_audit_sha256': sha(audit_path),
          'script_sha256': sha(Path(__file__)), 'complete_receptions': 300, 'all_checks_passed': True,
          'groups': groups, 'model_quality_aggregated': False, 'generation_parameters_changed': False,
          'interpretation': 'CRC-failed but data-correct packets are payload/CRC disagreement, not a claim that the CRC implementation is wrong. Masked correct tokens measure exposure before generation, not observed reconstruction damage. Zero observed missed payload errors does not establish zero population miss probability.'}
out = run / 'stage55_complete_phy_mask_exposure.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'groups'}, indent=2))
