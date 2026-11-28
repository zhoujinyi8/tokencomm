"""Release fixed/random quality differences only for complete matched SNR groups."""
import hashlib
import json
from pathlib import Path

from paired_packetization_aggregation import paired_snr_summary

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
spec=read('stage60_execution_spec.json')
baseline=read('stage59_development_recovery_audit.json');current=read('stage60_randomized_recovery_audit.json')
assert baseline['all_checks_passed'] and baseline['complete'] and baseline['completed_cases']==300
assert sha(run/'stage59_development_recovery_audit.json')==spec['baseline_complete_audit_sha256']
assert current['all_checks_passed'] and current['plan_sha256']==sha(run/'stage60_execution_spec.json')
assert current['audit_script_sha256']==sha(root/'audit_stage60_randomized_recovery.py')
contract=read('stage60_paired_aggregation_contract.json')
assert contract['passed']==5 and contract['failed']==0
assert contract['helper_sha256']==sha(root/'paired_packetization_aggregation.py')
assert contract['test_script_sha256']==sha(root/'test_paired_packetization_aggregation.py')
groups={}
for si,snr in enumerate(spec['snr_db']):
    if str(snr) not in current['complete_snr_groups']:continue
    cohorts=[]
    for namespace,audit in [('stage54_development_recovery',baseline),('stage60_randomized_recovery',current)]:
        rows=[]
        for name,digest in audit['case_sha256'].items():
            if not name.startswith(f'snr{si}_'):continue
            path=run/namespace/name;assert sha(path)==digest
            case=json.loads(path.read_text(encoding='utf-8'))
            assert sha(path.with_suffix('.npz'))==case['latent_sha256']
            rows.append(case)
        cohorts.append(rows)
    groups[str(snr)]=paired_snr_summary(*cohorts)
result=dict(all_checks_passed=True,audit_script_sha256=sha(Path(__file__)),
    plan_sha256=sha(run/'stage60_execution_spec.json'),preregistered_plan_sha256=spec['preregistered_plan_sha256'],
    baseline_audit_sha256=spec['baseline_complete_audit_sha256'],
    randomized_audit_sha256=sha(run/'stage60_randomized_recovery_audit.json'),
    randomized_cases=current['completed_cases'],randomized_counts_by_snr=current['counts_by_snr'],
    complete_randomized_study=current['complete'],paired_quality_snr_groups=len(groups),groups=groups,
    helper_sha256=sha(root/'paired_packetization_aggregation.py'),selected_pipeline=None,
    incomplete_snr_quality_means_withheld=True,new_model_inference_performed=False,primary_holdout_changed=False)
(run/'stage60_paired_packetization_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
