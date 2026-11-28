"""Freeze a separate fixed-budget proxy fine-tuning pilot, not an author checkpoint reproduction."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
smoke=json.loads((run/'stage65_gradient_smoke.json').read_text(encoding='utf-8'));assert smoke['all_checks_passed'] and smoke['optimizer_updates']==1
base=json.loads((run/'stage65_gradient_smoke_plan.json').read_text(encoding='utf-8'))
for p,h in base['evaluation_manifest_sha256'].items():assert sha(r/p)==h
forbidden=set();pixels=set();sources=set()
for p in base['evaluation_manifest_sha256']:
    for line in (r/p).read_text(encoding='utf-8').splitlines():
        v=json.loads(line)
        for k in ('sha256','image_sha256'):
            if v.get(k):forbidden.add(v[k])
        if v.get('pixel_sha256'):pixels.add(v['pixel_sha256'])
        if v.get('source_path'):sources.add(v['source_path'])
train=r/'data/imagenette_stage5/train.jsonl';assert sha(train)==base['training_manifest_sha256']
records=[json.loads(v) for v in train.read_text(encoding='utf-8').splitlines()]
eligible=[v for v in records if v['sha256'] not in forbidden and v.get('pixel_sha256') not in pixels and v.get('source_path') not in sources]
by_class={i:sorted([v for v in eligible if v['label']==i],key=lambda x:x['sha256'])[:10] for i in range(10)}
assert all(len(v)==10 for v in by_class.values())
selected=[by_class[c][j] for j in range(10) for c in range(10)]
images=[dict(path='data/imagenette_stage5/'+v['image'],sha256=v['sha256'],label=v['label'],pixel_sha256=v.get('pixel_sha256'),source_path=v.get('source_path')) for v in selected]
for v in images:assert sha(r/v['path'])==v['sha256']
diagnostic=json.loads((run/'stage65_cheng_diagnostic_plan.json').read_text(encoding='utf-8'))
assert not {v['sha256'] for v in images}&{v['sha256'] for v in diagnostic['images']}
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Independent ten-class proxy rate-calibration pilot, not the paper fine-tuned Cheng baseline',
    model='cheng2020-anchor',stock_weight_sha256=base['stock_weight_sha256'],images=images,
    selection='10 minimum-SHA training images per class, interleaved class order; fixed 100 updates, no checkpoint selection',
    training_manifest_sha256=sha(train),evaluation_manifest_sha256=base['evaluation_manifest_sha256'],
    calibration_images=diagnostic['images'],seed=2026112802,steps=100,batch_size=1,lmbda=.0001,learning_rate=.0001,aux_learning_rate=.001,
    max_grad_norm=1.,torch_threads=4,image_size=[256,256],target_entropy_bpp=.04,diagnostic_relative_tolerance=.1,
    initialization='original public anchor-q1 checkpoint; NOT the one-step smoke checkpoint',
    loss='upstream RateDistortionLoss(mse), lambda*255^2*MSE + estimated_bpp',
    calibration='All ten old development images exactly once after 100 updates; actual complete entropy strings, not proxy likelihoods',
    gradient_smoke_sha256=sha(run/'stage65_gradient_smoke.json'),worker_sha256=sha(r/'run_stage65_rate_pilot.py'),
    runtime_helper_sha256=sha(r/'cheng_runtime.py'),container_helper_sha256=sha(r/'cheng_stream_container.py'),
    author_training_hyperparameters_established=False,paper_dataset_identity_established=False,
    frozen_evaluation_images_used_for_training=False,final_test_performed=False,paper_baseline_ready=False)
p=run/'stage65_rate_pilot_plan.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(training_images=100,updates=100,plan_sha256=sha(p))))
