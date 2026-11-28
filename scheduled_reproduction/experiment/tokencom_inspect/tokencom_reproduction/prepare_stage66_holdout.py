"""Freeze the development-selected codec before evaluating the existing 100-image holdout."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
grid=read('stage66_rate_grid.json');audit=read('stage66_rate_grid_audit.json');base=read('stage66_rate_grid_plan.json')
assert audit['all_checks_passed'] and audit['complete'] and audit['source_result_sha256']==sha(run/'stage66_rate_grid.json')
assert audit['selection']['selected_arm_id'] is not None
arm=next(a for a in grid['arms'] if a['arm_id']==audit['selection']['selected_arm_id'])
assert sha(r/arm['checkpoint_path'])==arm['checkpoint_sha256']
data=r/'data/imagenette_stage44_holdout';manifest=data/'test.jsonl';assert sha(manifest)==base['evaluation_manifest_sha256']['data/imagenette_stage44_holdout/test.jsonl']
records=[json.loads(v) for v in manifest.read_text(encoding='utf-8').splitlines()]
assert len(records)==len({v['sha256'] for v in records})==100
images=[dict(image_index=i,path=str((data/v['image']).relative_to(r)),sha256=v['sha256'],pixel_sha256=v['pixel_sha256'],source_path=v['source_path'],label=v['label']) for i,v in enumerate(records)]
for v in images:assert sha(r/v['path'])==v['sha256']
for key in ('sha256','pixel_sha256','source_path'):
    assert not {v[key] for v in images}&{v[key] for v in base['images']}
assert not {v['sha256'] for v in images}&{v['sha256'] for v in base['calibration_images']}
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=66,
    scope='Independent clean Cheng endpoint on existing frozen ten-class 100-image holdout; not ImageNet100 paper reproduction',
    selected_arm_id=arm['arm_id'],lmbda=arm['lmbda'],checkpoint_path=arm['checkpoint_path'],checkpoint_sha256=arm['checkpoint_sha256'],
    grid_plan_sha256=sha(run/'stage66_rate_grid_plan.json'),grid_audit_sha256=sha(run/'stage66_rate_grid_audit.json'),
    selection=audit['selection'],images=images,expected_cases=100,holdout_manifest_sha256=sha(manifest),
    historical_evidence_sha256=base['historical_evidence_sha256'],image_size=[256,256],torch_threads=4,
    metric_lock_sha256=sha(run/'stage47_metrics_lock.json'),
    source_sha256={n:sha(r/n) for n in ('run_stage66_holdout.py','audit_stage66_holdout.py','cheng_runtime.py','cheng_stream_container.py','cheng_packet_budget.py')},
    fixed_preprocess='RGB, PIL bilinear Resize(256,256), ToTensor [0,1]',
    metrics='Actual decoded clipped [0,1] PSNR, normalize=True LPIPS Alex, PIL-clipped ViT-B/32 CLIP cosine; equal image weights',
    no_training=True,no_checkpoint_reselection=True,no_noisy_channel=True,no_retransmissions=True,
    primary_tokcom_protocol_unchanged=True,paper_dataset_identity_established=False,paper_baseline_ready=False,
    rate_interpretation='Report held-out actual rate as observed even outside development acceptance window; never retune on holdout',
    fresh_decoder_image_indices=[0,24,49,74,99],full_paper_reproduction_complete=False)
out=run/'stage66_holdout_plan.json';assert not out.exists();out.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(plan_sha256=sha(out),checkpoint_sha256=arm['checkpoint_sha256'],selected_arm_id=arm['arm_id'],images=100)))
