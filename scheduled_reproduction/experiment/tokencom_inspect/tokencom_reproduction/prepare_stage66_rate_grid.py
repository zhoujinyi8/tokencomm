"""Freeze three independent lambda arms before any new training."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;run=r/'runs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
old=read('stage65_rate_pilot_plan.json');audit=read('stage65_rate_pilot_audit.json')
assert audit['all_checks_passed'] and audit['complete'] and len(audit['fresh_decoder_checks'])==10
assert audit['source_result_sha256']==sha(run/'stage65_rate_pilot.json')
assert not audit['within_diagnostic_rate_tolerance'] and audit['mean_entropy_payload_bpp']<.036
for n,h in old['evaluation_manifest_sha256'].items():assert sha(r/n)==h
for v in old['images']+old['calibration_images']:assert sha(r/v['path'])==v['sha256']
assert sha(r/'data/imagenette_stage5/train.jsonl')==old['training_manifest_sha256']
old_receipt=read('stage65_protocol_gaps.json')
frozen={n:sha(run/n) for n in old_receipt['legacy_artifact_sha256_unchanged']}
assert frozen==old_receipt['legacy_artifact_sha256_unchanged']
for p in sorted(run.glob('stage65_*.json')):
    if 'published_local_sync' not in p.name and 'publication_verification' not in p.name:frozen[p.name]=sha(p)
plan={k:old[k] for k in ('images','calibration_images','stock_weight_sha256','training_manifest_sha256','evaluation_manifest_sha256','seed','batch_size','learning_rate','aux_learning_rate','max_grad_norm','torch_threads','image_size','target_entropy_bpp','diagnostic_relative_tolerance','loss')}
plan.update(created_utc=datetime.now(timezone.utc).isoformat(),stage=66,
    scope='Three-arm fixed-budget ten-class proxy development calibration; not author recipe or paper dataset',
    arms=[dict(arm_id=name,lmbda=value) for name,value in [('lambda_00015',.00015),('lambda_00025',.00025),('lambda_00040',.0004)]],
    steps=100,initialization='each arm independently reloads original public anchor q1, zero optimizer state, same seed/order',
    rationale='Previous lambda=0.0001 undershot 0.036 bpp; increase distortion weight on a fixed preregistered grid. No result-dependent expansion or early stopping.',
    selection='Only after all 3 arms complete 100 updates and all ten calibration images: within [.036,.044] entropy bpp, smallest absolute deviation from .04; frozen arm order breaks exact ties. None if no arm passes.',
    prior_plan_sha256=sha(run/'stage65_rate_pilot_plan.json'),prior_audit_sha256=sha(run/'stage65_rate_pilot_audit.json'),
    historical_evidence_sha256=frozen,prior_checkpoint_sha256=audit['checkpoint_sha256'],
    source_sha256={n:sha(r/n) for n in ('run_stage66_rate_grid.py','audit_stage66_rate_grid.py','cheng_rate_selection.py','cheng_runtime.py','cheng_stream_container.py','cheng_packet_budget.py')},
    paper_dataset_identity_established=False,author_training_hyperparameters_established=False,
    frozen_evaluation_images_used_for_training=False,paper_baseline_ready=False,final_test_performed=False)
dest=run/'stage66_rate_grid_plan.json';assert not dest.exists()
dest.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(plan_sha256=sha(dest),arms=plan['arms'],updates_per_arm=100,training_images=len(plan['images']))))
