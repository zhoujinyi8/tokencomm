"""Cross-receipt integrity and scoped scalar results for the stage 66 publication."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_rate_grid_plan.json');grid=read('stage66_rate_grid.json');ga=read('stage66_rate_grid_audit.json')
assert ga['all_checks_passed'] and ga['complete'] and ga['source_result_sha256']==sha(run/'stage66_rate_grid.json')
for n,h in plan['historical_evidence_sha256'].items():assert sha(run/n)==h
assert plan['created_utc']<grid['recorded_utc']<read('stage66_holdout_plan.json')['created_utc']
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
assert ga['audit_script_sha256']==sha(r/'audit_stage66_rate_grid.py')
metrics=read('stage66_perceptual.json');lock=read('stage66_perceptual_lock.json')
assert metrics['complete'] and len(metrics['cases'])==30 and metrics['lock_sha256']==sha(run/'stage66_perceptual_lock.json')
assert lock['script_sha256']==sha(r/'evaluate_stage66_perceptual.py') and lock['grid_audit_sha256']==sha(run/'stage66_rate_grid_audit.json')
assert lock['grid_result_sha256']==sha(run/'stage66_rate_grid.json') and lock['metric_lock_sha256']==sha(run/'stage47_metrics_lock.json')
assert len({(v['arm_id'],v['image_index']) for v in metrics['cases']})==30
for v in metrics['cases']:
    arm=next(a for a in grid['arms'] if a['arm_id']==v['arm_id']);case=next(c for c in arm['calibration_cases'] if c['image_index']==v['image_index'])
    assert v['container_sha256']==case['container_sha256'] and v['checkpoint_sha256']==arm['checkpoint_sha256'] and v['decoded_pixel_sha256']==case['decoded_pixel_sha256']
    assert abs(v['psnr_decoded']-case['psnr_decoded'])<1e-4 and not v['model_encoder_called']
    assert all(math.isfinite(v[k]) for k in ('psnr_decoded','lpips_normalized','clip_clipped'))
hp=read('stage66_holdout_plan.json');ha=read('stage66_holdout_audit.json');progress=read('stage66_holdout_progress.json')
assert ha['all_checks_passed'] and ha['complete'] and ha['completed_cases']==100 and len(ha['fresh_decoder_checks'])==5
assert ha['script_sha256']==sha(r/'audit_stage66_holdout.py') and ha['progress_sha256']==sha(run/'stage66_holdout_progress.json')
assert ha['plan_sha256']==sha(run/'stage66_holdout_plan.json') and hp['checkpoint_sha256']==ha['checkpoint_sha256']==sha(r/hp['checkpoint_path'])
for n,h in hp['source_sha256'].items():assert sha(r/n)==h
holdout=[]
for name,h in ha['case_sha256'].items():
    path=run/'stage66_holdout_cases'/name;assert sha(path)==h;v=json.loads(path.read_text(encoding='utf-8'))
    holdout.append({k:v[k] for k in ('image_index','source_image_sha256','entropy_payload_bpp','total_stored_bpp','estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped','container_sha256','decoded_pixels_sha256','packet_budget')})
assert len(holdout)==100 and {v['image_index'] for v in holdout}==set(range(100))
for key,value in ha['means'].items():
    values=[v[key] if key in v else v['packet_budget'][key] for v in holdout]
    assert abs(math.fsum(values)/100-value)<1e-9
resource=read('stage66_resource_budget.json');assert resource['all_checks_passed'] and resource['holdout_audit_sha256']==sha(run/'stage66_holdout_audit.json')
assert resource['script_sha256']==sha(r/'audit_stage66_resource_budget.py')
contracts=read('stage66_contracts.json');assert contracts['new_pure_checks']==17 and contracts['total_pure_ci_checks']==60 and contracts['failed']==0
assert contracts['script_sha256']==sha(r/'record_stage66_contracts.py') and contracts['helper_sha256']==sha(r/'cheng_rate_selection.py') and contracts['test_sha256']==sha(r/'test_cheng_rate_selection.py')
for n,h in contracts['transport_source_sha256'].items():assert sha(r/n)==h
paper=read('stage66_paper_version_audit.json');assert paper['all_checks_passed'] and paper['case_study_section_identical'] and paper['script_sha256']==sha(r/'audit_stage66_paper_versions.py')
assert paper['fetch_receipt_sha256']==sha(run/'stage66_paper_source_fetch.json')
journal=read('stage66_journal_extract_audit.json');assert journal['all_checks_passed'] and journal['focused_numerical_agreement_established']
assert journal['script_sha256']==sha(r/'audit_stage66_journal_extract.py')
ta=read('stage66_transport_audit.json');tp=read('stage66_transport_plan.json');assert ta['all_checks_passed'] and ta['progress_sha256']==sha(run/'stage66_transport_progress.json')
assert ta['script_sha256']==sha(r/'audit_stage66_transport.py')
for n,h in tp['source_sha256'].items():assert sha(r/n)==h
pilot=read('stage66_transport_pilot_audit.json');assert pilot['all_checks_passed'] and pilot['completed_frames']==24
for name,h in pilot['case_sha256'].items():assert ta['case_sha256'][name]==h==sha(run/'stage66_transport_cases'/name)
for path in (run/'stage66_transport_cases').glob('*.json'):
    v=json.loads(path.read_text(encoding='utf-8'));assert sha(path.with_suffix('.npz'))==v['trace_arrays_sha256']
endpoint_summary=None
if ta['complete']:
    ep=read('stage66_transport_endpoints.json');endpoint_summary=read('stage66_transport_endpoints_summary.json')
    assert ep['all_checks_passed'] and ep['transport_audit_sha256']==sha(run/'stage66_transport_audit.json')
    assert ep['script_sha256']==sha(r/'evaluate_stage66_transport_endpoints.py') and ep['new_decoder_forwards']==0
    assert endpoint_summary['detailed_rows_sha256']==sha(run/'stage66_transport_endpoints.json')
    assert {k:v for k,v in endpoint_summary.items() if k!='detailed_rows_sha256'}=={k:v for k,v in ep.items() if k!='rows'}
    assert len(ep['rows'])==900 and ep['reference_rows']+ep['unresolved_rows']==900
    ref={(v['arm_id'],v['image_index']):v for v in metrics['cases']}
    for row in ep['rows']:
        path=run/'stage66_transport_cases'/f"snr{tp['snr_db'].index(row['snr_db'])}_image{row['image_index']:02d}_trial{row['trial']}.json"
        frame=json.loads(path.read_text(encoding='utf-8'));assert sha(path)==row['source_frame_sha256']
        exact=frame['policies'][row['policy']]['complete_and_exact']
        assert (row['metrics'] is not None)==exact
        if exact:
            clean=ref[(tp['selected_arm_id'],row['image_index'])]
            assert row['metrics']=={k:clean[k] for k in ('psnr_decoded','lpips_normalized','clip_clipped')}
evidence_names=['stage66_rate_grid_plan.json','stage66_rate_grid.json','stage66_rate_grid_audit.json','stage66_perceptual_lock.json','stage66_perceptual.json',
    'stage66_holdout_plan.json','stage66_holdout_progress.json','stage66_holdout_audit.json','stage66_resource_budget.json','stage66_contracts.json',
    'stage66_paper_source_fetch.json','stage66_paper_version_audit.json','stage66_journal_extract_audit.json','stage66_remote_source_check.json','stage66_transport_contract.json','stage66_transport_plan.json',
    'stage66_transport_noiseless.json','stage66_transport_progress.json','stage66_transport_audit.json','stage66_transport_pilot_progress.json','stage66_transport_pilot_audit.json']
if ta['complete']:evidence_names.append('stage66_transport_endpoints_summary.json')
probe_audit=read('stage66_decode_probes_audit.json');assert probe_audit['all_checks_passed'] and probe_audit['script_sha256']==sha(r/'audit_stage66_decode_probes.py')
for v in probe_audit['probes']:
    assert v['plan_sha256']==sha(run/(v['probe']+'_plan.json')) and v['result_sha256']==sha(run/(v['probe']+'.json'))
completion=read('stage66_crc_endpoint_completion.json');assert completion['all_checks_passed'] and completion['complete_crc_endpoint_frames']==300
assert completion['script_sha256']==sha(r/'complete_stage66_crc_endpoints.py') and completion['source_endpoint_rows_sha256']==sha(run/'stage66_transport_endpoints.json')
assert completion['measured_corrupt_metric_sha256']==sha(run/'stage66_crc_miss_metrics.json')
deviation=read('stage66_crc_metric_preflight_deviation.json');assert deviation['all_checks_passed'] and not deviation['identity_gate_relaxed']
assert deviation['failed_preflight_source_sha256']==sha(run/'stage66_crc_miss_metrics_preflight_source.py')
evidence_names+=['stage66_received_headers_summary.json','stage66_decode_probe_plan.json','stage66_decode_probe.json','stage66_crc_miss_probe_plan.json',
    'stage66_crc_miss_probe.json','stage66_decode_probes_audit.json','stage66_crc_repeat_diagnostic.json','stage66_crc_miss_metrics_lock.json',
    'stage66_crc_miss_metrics_lock_v2.json','stage66_crc_miss_metrics.json','stage66_crc_metric_preflight_deviation.json','stage66_crc_endpoint_completion.json']
value=dict(all_checks_passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),stage=66,
    source_receipt_sha256={n:sha(run/n) for n in evidence_names},historical_artifacts_unchanged=len(plan['historical_evidence_sha256']),
    grid_updates=sum(a['updates_completed'] for a in grid['arms']),grid_fresh_decoder_checks=sum(len(a['fresh_decoder_checks']) for a in ga['arms']),
    selected_arm_id=ga['selection']['selected_arm_id'],selected_checkpoint_sha256=hp['checkpoint_sha256'],
    grid_selection=ga['selection'],grid_metric_means={a['arm_id']:{k:math.fsum(v[k] for v in metrics['cases'] if v['arm_id']==a['arm_id'])/10 for k in ('psnr_decoded','lpips_normalized','clip_clipped')} for a in grid['arms']},
    holdout_cases=100,holdout_fresh_decoder_checks=5,holdout_means=ha['means'],holdout_per_image=sorted(holdout,key=lambda v:v['image_index']),
    holdout_entropy_rate_relative_difference_from_target=ha['means']['entropy_payload_bpp']/.04-1,
    holdout_within_diagnostic_window=abs(ha['means']['entropy_payload_bpp']-.04)<=.004,
    transport_frames=ta['completed_frames'],transport_complete=ta['complete'],transport_complete_snr_groups=len(ta['complete_snr_groups']),
    transport_endpoint_summary=endpoint_summary,
    crc_endpoint_complete_frames=300,crc_clean_reference_rows=299,crc_actual_corrupt_metric_rows=1,
    received_decode_probe_coordinates=7,corrupted_crc_psnr=read('stage66_crc_miss_metrics.json')['psnr_decoded'],
    initial_24_transport_cases_unchanged=True,new_pure_contract_checks=17,total_pure_ci_contract_checks=60,
    no_holdout_training_or_reselection=True,paper_dataset_identity_established=False,paper_baseline_ready=False,
    full_resource_equivalence_established=False,full_paper_reproduction_complete=False)
(run/'stage66_session_audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in value.items() if k not in ('holdout_per_image','source_receipt_sha256')},indent=2))
