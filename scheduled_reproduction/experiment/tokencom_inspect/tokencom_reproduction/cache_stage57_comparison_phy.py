"""Cache only the new token stream under the preregistered matched 8 dB comparison."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from smoke_stage25_sionna_link import simulate

root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: json.loads((run/n).read_text(encoding='utf-8'))
plan_path = run/'stage56_end_to_end_convention_plan.json'
plan = read(plan_path.name)
range_audit = read('stage56_range_audit.json')
source_phy_audit = read('stage54_development_phy_audit.json')
assert range_audit['complete'] and range_audit['all_checks_passed']
assert source_phy_audit['complete'] and source_phy_audit['all_checks_passed']
assert source_phy_audit['plan_sha256'] == plan['source_multisnr_plan_sha256']
assert range_audit['plan_sha256'] == plan['source_range_plan_sha256']
assert sha(root/'smoke_stage25_sionna_link.py') == read('stage54_development_multisnr_plan.json')['phy_simulator_sha256']
identity = {'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
            'range_audit_sha256': sha(run/'stage56_range_audit.json'),
            'source_phy_audit_sha256': sha(run/'stage54_development_phy_audit.json'),
            'simulator_sha256': sha(root/'smoke_stage25_sionna_link.py')}
lock_path = run/'stage57_comparison_phy_lock.json'
if lock_path.exists():
    assert read(lock_path.name) == identity
else:
    lock_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
target = run/'stage57_comparison_phy'
target.mkdir(exist_ok=True)
for image in range(10):
    case_path = run/f'stage56_range_cases/image{image:02d}.json'
    case = json.loads(case_path.read_text(encoding='utf-8'))
    assert sha(case_path) == range_audit['case_sha256'][case_path.name]
    assert sha(case_path.with_suffix('.npz')) == case['arrays_sha256']
    with np.load(case_path.with_suffix('.npz')) as arrays:
        tokens = arrays['training_tokens'].copy()
    for trial in range(5):
        demo = run/f'stage54_development_phy/snr4_image{image:02d}_trial{trial}.npz'
        assert sha(demo) == source_phy_audit['case_sha256'][demo.name]
        out = target/f'image{image:02d}_trial{trial}.npz'
        if out.exists():
            with np.load(out) as saved:
                assert saved['lock_sha256'].item() == sha(lock_path)
                assert np.array_equal(saved['transmitted'], tokens)
            continue
        seed = plan['phy_seed_base']+100*image+trial
        metadata, received, oracle, crc = simulate(seed, plan['snr_db'], tokens=tokens, return_arrays=True)
        temporary = out.with_suffix('.partial.npz')
        np.savez_compressed(temporary, transmitted=tokens, received=received.reshape(256),
            oracle_token_mask=np.tile(oracle, 4), crc_token_mask=np.tile(crc, 4),
            source_sha256=plan['source_image_manifest'][image]['image_sha256'],
            lock_sha256=sha(lock_path), plan_sha256=sha(plan_path), metadata=json.dumps(metadata))
        os.replace(temporary, out)
        print(f'comparison PHY image={image+1}/10 trial={trial+1}/5', flush=True)
files = sorted(target.glob('image[0-9][0-9]_trial[0-4].npz'))
result = {'plan_sha256': sha(plan_path), 'lock_sha256': sha(lock_path), 'complete': len(files) == 50,
          'completed_training_convention_cases': len(files), 'demo_reference_cases': 50,
          'case_sha256': {p.name: sha(p) for p in files}, 'primary_holdout_changed': False}
(run/'stage57_comparison_phy_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'case_sha256'}, indent=2))
