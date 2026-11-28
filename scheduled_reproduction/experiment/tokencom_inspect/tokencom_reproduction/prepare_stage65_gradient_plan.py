"""Separate one-update training smoke plan; it is not the clean-codec diagnostic or paper fine-tuning."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;data=r/'data';run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
forbidden_bytes=set();forbidden_pixels=set();forbidden_sources=set();manifests={};rows_count=0
for p in sorted(data.rglob('*.jsonl')):
    if not (p.name.startswith('val') or p.name.startswith('test')):continue
    manifests[p.relative_to(r).as_posix()]=sha(p)
    for line in p.read_text(encoding='utf-8').splitlines():
        row=json.loads(line);rows_count+=1
        if row.get('sha256'):forbidden_bytes.add(row['sha256'])
        if row.get('image_sha256'):forbidden_bytes.add(row['image_sha256'])
        if row.get('pixel_sha256'):forbidden_pixels.add(row['pixel_sha256'])
        if row.get('source_path'):forbidden_sources.add(row['source_path'])
train=data/'imagenette_stage5/train.jsonl';records=[json.loads(s) for s in train.read_text(encoding='utf-8').splitlines()]
valid=[v for v in records if v['sha256'] not in forbidden_bytes and v.get('pixel_sha256') not in forbidden_pixels and v.get('source_path') not in forbidden_sources]
selected=sorted(valid,key=lambda v:v['sha256'])[0];image=data/'imagenette_stage5'/selected['image'];assert sha(image)==selected['sha256']
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='One optimizer-update engineering smoke, not paper fine-tuning or rate-matched evaluation',
    model='cheng2020-anchor',stock_weight_sha256='dad2ebff44e86ae986d464d286d4c3531bd0f8c038233601bb8f3ad6dc7754e7',
    image=dict(path=image.relative_to(r).as_posix(),**selected),selection_rule='minimum SHA256 among eligible existing training rows',
    training_manifest_sha256=sha(train),evaluation_manifest_sha256=manifests,evaluation_rows_checked=rows_count,
    excluded_identity_counts=dict(byte_sha256=len(forbidden_bytes),pixel_sha256=len(forbidden_pixels),source_path=len(forbidden_sources)),
    eligible_training_rows=len(valid),seed=2026112801,torch_threads=4,steps=1,batch_size=1,image_size=[256,256],
    loss='upstream RateDistortionLoss(mse): lambda*255^2*MSE + estimated_bpp',lmbda=.0001,
    optimizer='Adam',learning_rate=.00001,aux_learning_rate=.001,max_grad_norm=1.,
    hyperparameters_are_engineering_choices=True,author_training_hyperparameters_established=False,
    expected_contracts=['finite loss/gradients','nonzero learned parameter updates','updated entropy CDFs','checkpoint reload','decoded pixels equal after reload'],
    stock_weights_will_be_overwritten=False,frozen_evaluation_images_used_for_training=False,
    source_backend_sha256=sha(run/'stage65_backend_smoke.json'),paper_baseline_ready=False,rate_match_established=False)
p=run/'stage65_gradient_smoke_plan.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(plan_sha256=sha(p),eligible_training_rows=len(valid),evaluation_rows_checked=rows_count,selected_image_sha256=selected['sha256'])))
