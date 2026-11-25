"""Validate both complete PHY streams before matched-pipeline reconstruction."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: json.loads((run/n).read_text(encoding='utf-8'))
plan = read('stage56_end_to_end_convention_plan.json')
progress = read('stage57_comparison_phy_progress.json')
lock = read('stage57_comparison_phy_lock.json')
range_audit = read('stage56_range_audit.json')
old = read('stage54_development_phy_audit.json')
assert progress['complete'] and progress['completed_training_convention_cases'] == 50
assert progress['plan_sha256'] == lock['plan_sha256'] == sha(run/'stage56_end_to_end_convention_plan.json')
assert progress['lock_sha256'] == sha(run/'stage57_comparison_phy_lock.json')
assert lock['script_sha256'] == sha(root/'cache_stage57_comparison_phy.py')
assert lock['range_audit_sha256'] == sha(run/'stage56_range_audit.json')
assert lock['source_phy_audit_sha256'] == sha(run/'stage54_development_phy_audit.json')
cases = []
for image in range(10):
    source_json = run/f'stage56_range_cases/image{image:02d}.json'
    source = json.loads(source_json.read_text(encoding='utf-8'))
    assert sha(source_json) == range_audit['case_sha256'][source_json.name]
    assert sha(source_json.with_suffix('.npz')) == source['arrays_sha256']
    with np.load(source_json.with_suffix('.npz')) as arrays:
        tokens = {'demo': arrays['demo_tokens'].copy(), 'training_code_convention': arrays['training_tokens'].copy()}
    for trial in range(5):
        pair = {'image_index': image, 'trial': trial, 'source_sha256': source['source_sha256'], 'pipelines': {}}
        for pipeline in ('demo', 'training_code_convention'):
            path = (run/f'stage54_development_phy/snr4_image{image:02d}_trial{trial}.npz'
                    if pipeline == 'demo' else run/f'stage57_comparison_phy/image{image:02d}_trial{trial}.npz')
            assert sha(path) == (old['case_sha256'][path.name] if pipeline == 'demo' else progress['case_sha256'][path.name])
            with np.load(path) as saved:
                rx = saved['received']; tx = tokens[pipeline]
                assert rx.shape == tx.shape == (256,) and rx.min() >= 0 and rx.max() < 1024
                if pipeline != 'demo':
                    assert np.array_equal(saved['transmitted'], tx)
                    assert saved['lock_sha256'].item() == progress['lock_sha256']
                assert saved['source_sha256'].item() == source['source_sha256']
                wrong = tx != rx
                bad = wrong.reshape(4, 64).any(axis=0)
                oracle = saved['oracle_token_mask'].astype(bool).reshape(4, 64)
                crc = saved['crc_token_mask'].astype(bool).reshape(4, 64)
                assert np.array_equal(oracle, np.broadcast_to(bad, (4, 64)))
                assert np.array_equal(crc, np.broadcast_to(crc[0], (4, 64)))
                metadata = json.loads(saved['metadata'].item())
                assert metadata['seed'] == plan['phy_seed_base']+100*image+trial
                assert metadata['snr_db'] == 8.0
                assert metadata['raw_bits_per_packet'] == 40 and metadata['crc_bits_per_packet'] == 51
                assert metadata['coded_bits_per_packet'] == 102 and metadata['qam_symbols_total'] == 1632
                bit_errors = int((((tx.astype(np.int64)^rx.astype(np.int64))[:, None] & (1 << np.arange(10))) != 0).sum())
                assert metadata['bit_errors'] == bit_errors and metadata['token_errors'] == int(wrong.sum())
                assert metadata['oracle_bad_packets'] == int(bad.sum())
                assert metadata['crc_bad_packets'] == int(crc[0].sum())
                pair['pipelines'][pipeline] = {'phy_sha256': sha(path), 'true_error_packets': int(bad.sum()),
                    'wrong_tokens': int(wrong.sum()), 'crc_failed_packets': int(crc[0].sum()),
                    'crc_failed_data_correct_packets': int((crc[0] & ~bad).sum())}
        cases.append(pair)
result = {'plan_sha256': progress['plan_sha256'], 'all_checks_passed': True, 'complete': True,
          'complete_pairs': 50, 'pipeline_receptions': 100, 'new_phy_simulations': 50, 'demo_phy_reused': 50,
          'cases': cases, 'primary_holdout_changed': False, 'model_quality_aggregated': False}
(run/'stage57_comparison_phy_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
