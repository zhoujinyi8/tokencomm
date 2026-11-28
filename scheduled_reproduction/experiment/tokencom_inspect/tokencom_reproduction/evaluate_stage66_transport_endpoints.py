"""Reuse frozen clean image metrics only for byte-identical, uncensored transport endpoints."""
import hashlib,json,math
from pathlib import Path
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
audit=read('stage66_transport_audit.json');plan=read('stage66_transport_plan.json');metrics=read('stage66_perceptual.json');grid_audit=read('stage66_rate_grid_audit.json');grid=read('stage66_rate_grid.json')
assert audit['all_checks_passed'] and audit['complete'] and audit['completed_frames']==300
assert audit['progress_sha256']==sha(run/'stage66_transport_progress.json')
assert metrics['complete'] and metrics['lock_sha256']==sha(run/'stage66_perceptual_lock.json')
lock=read('stage66_perceptual_lock.json');assert lock['grid_audit_sha256']==sha(run/'stage66_rate_grid_audit.json') and lock['grid_result_sha256']==sha(run/'stage66_rate_grid.json')
assert plan['grid_audit_sha256']==sha(run/'stage66_rate_grid_audit.json')
selected=grid_audit['selection']['selected_arm_id'];assert selected==plan['selected_arm_id']
arm=next(a for a in grid['arms'] if a['arm_id']==selected);assert arm['checkpoint_sha256']==plan['checkpoint_sha256']
reference={v['image_index']:v for v in metrics['cases'] if v['arm_id']==selected};assert len(reference)==10
rows=[]
for name,h in audit['case_sha256'].items():
    path=run/'stage66_transport_cases'/name;assert sha(path)==h;frame=json.loads(path.read_text(encoding='utf-8'))
    ref=reference[frame['image_index']];source=next(v for v in plan['streams'] if v['image_index']==frame['image_index'])
    assert ref['checkpoint_sha256']==plan['checkpoint_sha256'] and ref['container_sha256']==frame['source_container_sha256']==source['container_sha256']
    for policy,v in frame['policies'].items():
        exact=v['complete_and_exact'];assert exact==(v['container_exact'] and v['censored_packets']==0)
        values={k:ref[k] for k in ('psnr_decoded','lpips_normalized','clip_clipped')} if exact else None
        rows.append(dict(image_index=frame['image_index'],trial=frame['trial'],snr_db=frame['snr_db'],policy=policy,
            source_frame_sha256=h,source_container_sha256=ref['container_sha256'],checkpoint_sha256=ref['checkpoint_sha256'],
            endpoint='exact_byte_clean_reference' if exact else 'unresolved_corrupt_or_censored_stream',metrics=values,new_decoder_forwards=0))
assert len(rows)==len({(v['image_index'],v['trial'],v['snr_db'],v['policy']) for v in rows})==900
groups={}
for snr in plan['snr_db']:
    policies={}
    for policy in ('single_pass_padded','receiver_crc_arq_padded','sender_oracle_arq_padded'):
        subset=[v for v in rows if v['snr_db']==snr and v['policy']==policy];assert len(subset)==50
        known=[v for v in subset if v['metrics'] is not None]
        policies[policy]=dict(frames=50,exact_byte_reference_rows=len(known),unresolved_rows=50-len(known),
            means={k:math.fsum(v['metrics'][k] for v in known)/50 for k in ('psnr_decoded','lpips_normalized','clip_clipped')} if len(known)==50 else None)
    groups[str(snr)]=policies
result=dict(all_checks_passed=True,script_sha256=sha(__file__),transport_audit_sha256=sha(run/'stage66_transport_audit.json'),
    perceptual_result_sha256=sha(run/'stage66_perceptual.json'),grid_audit_sha256=sha(run/'stage66_rate_grid_audit.json'),
    reference_rows=sum(v['metrics'] is not None for v in rows),unresolved_rows=sum(v['metrics'] is None for v in rows),
    new_decoder_forwards=0,received_corrupt_streams_decoded=0,per_snr=groups,rows=rows,
    interpretation='Exact byte equality and uncensored stopping justify reusing the same frozen clean decode. No metrics are imputed for other streams and no conditional-only mean is labeled a full-cohort result.',
    paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage66_transport_endpoints.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
summary={k:v for k,v in result.items() if k!='rows'}
summary['detailed_rows_sha256']=sha(run/'stage66_transport_endpoints.json')
(run/'stage66_transport_endpoints_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
