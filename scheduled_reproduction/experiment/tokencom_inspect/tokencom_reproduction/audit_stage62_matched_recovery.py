"""Validate pilot outputs and summarize only complete preplanned SNR groups."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from matched_recovery_checks import validate

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
plan_path=run/'stage61_matched_padded_recovery_plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
lock_path=run/'stage62_matched_recovery_lock.json';lock=json.loads(lock_path.read_text(encoding='utf-8'))
progress=json.loads((run/'stage62_matched_recovery_progress.json').read_text(encoding='utf-8'))
phy_audit=json.loads((run/'stage61_matched_padded_input_audit.json').read_text(encoding='utf-8'))
assert lock['script_sha256']==sha(root/'run_stage62_matched_recovery.py')
assert lock['checks_sha256']==sha(root/'matched_recovery_checks.py')
assert lock['test_sha256']==sha(root/'test_matched_recovery_checks.py')
assert lock['auditor_sha256']==sha(Path(__file__))
assert lock['plan_sha256']==progress['plan_sha256']==phy_audit['plan_sha256']==sha(plan_path)
assert progress['lock_sha256']==sha(lock_path) and phy_audit['all_checks_passed']
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as link:sent=link['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as source:images=json.loads(source['metadata'].item())['rows']
paths=plan['paths'];keys=['correct_tokens','whole_image_ter','psnr_raw','lpips_normalized','clip_clipped']
cases=[];hashes={}
for path in sorted((run/'stage62_matched_recovery').glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json')):
    case=json.loads(path.read_text(encoding='utf-8'))
    match=re.fullmatch(r'snr([0-5])_image([0-9]{2})_trial([0-4])',path.stem)
    si,image,trial=map(int,match.groups());assert image<10
    assert (case['snr_index'],case['image_index'],case['trial'])==(si,image,trial)
    assert case['snr_db']==plan['snr_db'][si] and case['source_sha256']==images[image]['image_sha256']
    assert sha(path)==progress['case_sha256'][path.name] and case['lock_sha256']==sha(lock_path)
    phy_path=run/'stage61_matched_padded_inputs'/f'{path.stem}.npz'
    latent_path=path.with_suffix('.npz')
    assert case['source_phy_sha256']==sha(phy_path)==phy_audit['case_sha256'][phy_path.name]
    assert case['latent_sha256']==sha(latent_path)
    assert [row['path'] for row in case['rows']]==paths
    with np.load(phy_path) as phy,np.load(latent_path) as latent:
        mask=phy['oracle_token_mask'].astype(bool);rx=phy['received'];tx=sent[image]
        assert case['masked_tokens']=={'oracle':int(mask.sum()),'receiver_crc':int(phy['crc_token_mask'].sum())}
        validate(case['rows'],dict(latent),rx,tx,mask,phy['crc_token_mask'].astype(bool))
        assert np.array_equal(latent['direct'],rx) and np.array_equal(latent['ideal_correct_tokens'],tx)
        for row in case['rows']:
            ids=latent[row['path']]
            assert ids.shape==(256,) and ids.min()>=0 and ids.max()<1024
            assert row['correct_tokens']==int((ids==tx).sum())
            assert row['whole_image_ter']==1-row['correct_tokens']/256
            assert 0<=row['steps']<=32 and all(math.isfinite(row[key]) for key in keys)
            if row['path'] not in ('direct','ideal_correct_tokens'):
                use_mask=phy['crc_token_mask'].astype(bool) if row['path'].startswith('receiver_crc') else mask
                assert np.array_equal(ids[~use_mask],rx[~use_mask])
    cases.append(case);hashes[path.name]=sha(path)
assert len(cases)==progress['completed_cases']
coverage={};groups={}
for si,snr in enumerate(plan['snr_db']):
    subset=[case for case in cases if case['snr_index']==si]
    coverage[str(snr)]=len(subset)
    assert len({(c['image_index'],c['trial']) for c in subset})==len(subset)
    if len(subset)==50:
        assert {(c['image_index'],c['trial']) for c in subset}=={(i,t) for i in range(10) for t in range(5)}
        per_image=[{name:{key:sum(next(row for row in case['rows'] if row['path']==name)[key]
            for case in subset if case['image_index']==image)/5 for key in keys} for name in paths} for image in range(10)]
        means={name:{key:sum(row[name][key] for row in per_image)/10 for key in keys} for name in paths}
        assert math.isclose(means['ideal_correct_tokens']['correct_tokens'],256)
        groups[str(snr)]={'complete_receptions':50,'means_over_images':means,
            'per_image_trial_means':per_image,'forward_qam_symbols_per_frame':1664}
result={'scope':plan['scope'],'plan_sha256':sha(plan_path),'lock_sha256':sha(lock_path),
        'completed_cases':len(cases),'expected_cases':300,'complete':len(cases)==300,
        'counts_by_snr':coverage,'complete_snr_groups':groups,'all_checks_passed':True,'case_sha256':hashes,
        'primary_holdout_changed':False,'primary_summary_sha256':sha(run/'stage47_complete_holdout_summary.json'),
        'metric_definition':'actual final-ID equality and whole-image residual TER; author TER_af not used as TER',
        'physical_retries_simulated':False}
assert sha(run/'stage59_randomized_packetization_plan.json')==plan['source_randomization_plan_sha256']
assert lock['input_audit_sha256']==sha(run/'stage61_matched_padded_input_audit.json')
for name in ('runs/stage26_official_token_link_snr8_5trials.npz','runs/stage26_official_demo_tokens.npz'):
    assert sha(root/name)==plan['input_sha256'][name]
baseline=json.loads((run/'stage59_development_recovery_audit.json').read_text(encoding='utf-8'))
for case in cases:
    name=f"snr{case['snr_index']}_image{case['image_index']:02d}_trial{case['trial']}.json"
    original=run/'stage54_development_recovery'/name
    assert sha(original)==baseline['case_sha256'][name]
    expected=next(row for row in json.loads(original.read_text(encoding='utf-8'))['rows'] if row['path']=='ideal_correct_tokens')
    actual=next(row for row in case['rows'] if row['path']=='ideal_correct_tokens')
    for key,tolerance in [('psnr_raw',1e-4),('lpips_normalized',1e-5),('clip_clipped',1e-5)]:
        assert math.isclose(actual[key],expected[key],abs_tol=tolerance,rel_tol=0)
result.update(audit_script_sha256=sha(Path(__file__)),preregistered_plan_sha256=sha(plan_path),
              input_audit_sha256=sha(run/'stage61_matched_padded_input_audit.json'),
              ideal_reference_matches_fixed_baseline=True)
(run/'stage62_matched_recovery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('case_sha256','complete_snr_groups')},indent=2))
