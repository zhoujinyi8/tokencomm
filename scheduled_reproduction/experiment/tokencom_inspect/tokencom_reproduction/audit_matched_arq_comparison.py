"""Compare complete neural/ARQ groups with exact first-pass and metric anchors."""
import hashlib,json,math
from pathlib import Path
import numpy as np

KEYS=('correct_tokens','whole_image_ter','psnr_raw','lpips_normalized','clip_clipped')
def summarize(root):
    root=Path(root);run=root/'runs'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
    neural=read('stage62_matched_recovery_audit.json');metrics=read('stage61_arq_metric_audit.json')
    physical=read('stage61_packet_arq_audit.json');plan=read('stage61_matched_padded_recovery_plan.json')
    assert all(a['all_checks_passed'] for a in (neural,metrics,physical))
    assert metrics['complete'] and physical['complete']
    assert sha(run/'stage61_packet_arq_audit.json')==plan['source_arq_audit_sha256']==metrics['source_arq_audit_sha256']
    assert neural['audit_script_sha256']==sha(root/'audit_stage62_matched_recovery.py')
    assert metrics['audit_script_sha256']==sha(root/'audit_stage61_arq_metrics.py')
    assert physical['audit_script_sha256']==sha(root/'audit_stage61_packet_arq.py')
    nl=read('stage62_matched_recovery_lock.json');ml=read('stage61_arq_metric_lock.json')
    assert nl['metric_lock_sha256']==ml['metric_lock_sha256']
    assert nl['author_environment_sha256']==ml['author_environment_sha256']
    # Compare exact same source frame and actual decoded direct endpoint, including incomplete groups.
    for name,digest in neural['case_sha256'].items():
        p=run/'stage62_matched_recovery'/name;assert sha(p)==digest
        n=json.loads(p.read_text(encoding='utf-8'))
        ap=run/'stage61_arq_metrics'/name;assert sha(ap)==metrics['case_sha256'][name]
        a=json.loads(ap.read_text(encoding='utf-8'))
        inp=run/'stage61_matched_padded_inputs'/p.with_suffix('.npz').name
        assert sha(inp)==n['source_phy_sha256']
        with np.load(inp) as saved:
            assert saved['source_arq_case_sha256'].item()==a['source_arq_case_sha256']
            assert saved['source_trace_arrays_sha256'].item()==a['source_arq_arrays_sha256']
        direct=next(r for r in n['rows'] if r['path']=='direct')
        same=next(r for r in a['rows'] if r['path']=='single_pass_padded')
        for key in KEYS:
            tol=1e-4 if key=='psnr_raw' else 1e-5 if key in ('lpips_normalized','clip_clipped') else 0
            assert math.isclose(direct[key],same[key],rel_tol=0,abs_tol=tol)
    groups={}
    for snr,ng in neural['complete_snr_groups'].items():
        ag=metrics['complete_snr_groups'][snr];pg=physical['complete_snr_groups'][snr]
        assert ng['complete_receptions']==ag['complete_frames']==50
        paired={}
        for path in plan['paths']:
            paired[path]={}
            for policy in ('single_pass_padded','receiver_crc_arq_padded','sender_oracle_arq_padded'):
                values=[{k:ng['per_image_trial_means'][i][path][k]-ag['per_image_trial_means'][i][policy][k] for k in KEYS} for i in range(10)]
                paired[path][policy]=dict(per_image_differences=values,mean_difference={k:sum(v[k] for v in values)/10 for k in KEYS})
                for k in KEYS:
                    assert math.isclose(paired[path][policy]['mean_difference'][k],ng['means_over_images'][path][k]-ag['means_over_images'][policy][k],rel_tol=0,abs_tol=1e-10)
        groups[snr]=dict(neural_means=ng['means_over_images'],arq_means=ag['means_over_images'],
            neural_first_pass_symbols=1664,arq_forward_pixels_per_symbol={p:v['forward_pixels_per_symbol_ratio_of_sums'] for p,v in pg.items()},
            paired_neural_minus_arq=paired)
    result=dict(all_checks_passed=True,script_sha256=sha(Path(__file__)),neural_audit_sha256=sha(run/'stage62_matched_recovery_audit.json'),
        arq_metric_audit_sha256=sha(run/'stage61_arq_metric_audit.json'),physical_audit_sha256=sha(run/'stage61_packet_arq_audit.json'),
        anchor_cases_checked=neural['completed_cases'],complete_snr_groups=groups,complete=len(groups)==6,
        feedback_header_cmi_airtime_modeled=False,ideal_path_is_reference=True,full_paper_reproduction_complete=False)
    (run/'stage64_matched_arq_comparison.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
    return result
if __name__=='__main__':
    a=summarize(Path(__file__).resolve().parent)
    print(json.dumps({k:v for k,v in a.items() if k!='complete_snr_groups'}))
