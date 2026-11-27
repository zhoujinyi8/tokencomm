"""Audit endpoint metrics against physical ARQ traces and the frozen clean reference."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from arq_metric_checks import KEYS,validate_rows

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage61_arq_metric_plan.json');lock_path=run/'stage61_arq_metric_lock.json';lock=read(lock_path.name)
progress=read('stage61_arq_metric_progress.json');source=read('stage61_packet_arq_audit.json')
assert source['complete'] and source['all_checks_passed']
assert lock['script_sha256']==sha(root/'evaluate_stage61_arq_metrics.py')
assert lock['plan_sha256']==progress['plan_sha256']==sha(run/'stage61_arq_metric_plan.json')
assert lock['source_arq_audit_sha256']==sha(run/'stage61_packet_arq_audit.json')
assert sha(run/'stage59_development_recovery_audit.json')==plan['baseline_complete_audit_sha256']
contract=read('stage61_arq_metric_contract.json');assert contract['passed']==6 and contract['failed']==0
assert contract['helper_sha256']==sha(root/'arq_metric_checks.py')
assert contract['test_script_sha256']==sha(root/'test_arq_metric_checks.py')
for name in ('runs/stage26_official_token_link_snr8_5trials.npz','runs/stage26_official_demo_tokens.npz'):
    assert sha(root/name)==plan['input_sha256'][name]
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:images=json.loads(saved['metadata'].item())['rows']
baseline=read('stage59_development_recovery_audit.json');cases=[];hashes={}
for path in sorted((run/'stage61_arq_metrics').glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json')):
    si,image,trial=map(int,re.fullmatch(r'snr([0-5])_image([0-9]{2})_trial([0-4])',path.stem).groups())
    case=json.loads(path.read_text(encoding='utf-8'));assert sha(path)==progress['case_sha256'][path.name]
    assert (case['snr_index'],case['image_index'],case['trial'])==(si,image,trial) and case['snr_db']==plan['snr_db'][si]
    assert case['source_image_sha256']==images[image]['image_sha256'] and case['lock_sha256']==sha(lock_path)
    physical=run/'stage61_packet_arq'/path.name;assert sha(physical)==case['source_arq_case_sha256']==source['case_sha256'][path.name]
    physical_case=json.loads(physical.read_text(encoding='utf-8'))
    assert sha(physical.with_suffix('.npz'))==case['source_arq_arrays_sha256']==physical_case['trace_arrays_sha256']
    reference=run/'stage54_development_recovery'/path.name;assert sha(reference)==baseline['case_sha256'][path.name]
    clean=next(r for r in json.loads(reference.read_text(encoding='utf-8'))['rows'] if r['path']=='ideal_correct_tokens')
    with np.load(physical.with_suffix('.npz')) as saved:
        ids={policy:saved[f'final_{policy}'] for policy in plan['paths']}
        validate_rows(case['rows'],ids,sent[image],clean)
        for row in case['rows']:assert 256-row['correct_tokens']==physical_case['policies'][row['path']]['final_token_errors']
    cases.append(case);hashes[path.name]=sha(path)
assert len(cases)==progress['completed_frames'] and set(hashes)==set(progress['case_sha256'])
coverage={};groups={}
for si,snr in enumerate(plan['snr_db']):
    subset=[case for case in cases if case['snr_index']==si];coverage[str(snr)]=len(subset)
    assert len({(c['image_index'],c['trial']) for c in subset})==len(subset)
    if len(subset)==50:
        assert {(c['image_index'],c['trial']) for c in subset}=={(i,t) for i in range(10) for t in range(5)}
        per_image=[{p:{key:sum(next(r for r in c['rows'] if r['path']==p)[key] for c in subset if c['image_index']==i)/5
                       for key in KEYS} for p in plan['paths']} for i in range(10)]
        groups[str(snr)]=dict(complete_frames=50,means_over_images={p:{k:sum(row[p][k] for row in per_image)/10 for k in KEYS} for p in plan['paths']},per_image_trial_means=per_image)
result=dict(all_checks_passed=True,audit_script_sha256=sha(Path(__file__)),plan_sha256=lock['plan_sha256'],
    source_arq_audit_sha256=lock['source_arq_audit_sha256'],completed_frames=len(cases),expected_frames=300,
    complete=len(cases)==300,metric_rows=3*len(cases),counts_by_snr=coverage,complete_snr_groups=groups,
    exact_clean_reference_rows=sum(row['metric_source']=='exact_clean_id_reference' for c in cases for row in c['rows']),
    decoder_forwards_this_invocation=progress['decoder_forwards_this_invocation'],case_sha256=hashes,
    transformer_inference_performed=False,existing_studies_changed=False,full_paper_reproduction_complete=False)
(run/'stage61_arq_metric_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('case_sha256','complete_snr_groups')},indent=2))
