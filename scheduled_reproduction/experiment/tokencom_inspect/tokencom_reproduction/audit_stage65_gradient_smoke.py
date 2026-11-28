import json,hashlib
from pathlib import Path
import torch
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((run/'stage65_gradient_smoke_plan.json').read_text(encoding='utf-8'));a=json.loads((run/'stage65_gradient_smoke.json').read_text(encoding='utf-8'))
assert a['all_checks_passed'] and a['script_sha256']==sha(r/'run_stage65_gradient_smoke.py') and a['plan_sha256']==sha(run/'stage65_gradient_smoke_plan.json')
assert a['optimizer_updates']==plan['steps']==1 and a['changed_parameter_tensors']>0 and a['reloaded_decode_max_abs_difference']<=1e-6
for n,h in plan['evaluation_manifest_sha256'].items():assert sha(r/n)==h
assert sha(r/'data/imagenette_stage5/train.jsonl')==plan['training_manifest_sha256']
forbidden=set();pixels=set();sources=set()
for n in plan['evaluation_manifest_sha256']:
    for line in (r/n).read_text(encoding='utf-8').splitlines():
        v=json.loads(line)
        for k in ('sha256','image_sha256'):
            if v.get(k):forbidden.add(v[k])
        if v.get('pixel_sha256'):pixels.add(v['pixel_sha256'])
        if v.get('source_path'):sources.add(v['source_path'])
v=plan['image'];assert sha(r/v['path'])==a['image_sha256']==v['sha256'] and v['sha256'] not in forbidden and v['pixel_sha256'] not in pixels and v['source_path'] not in sources
stock=r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar';assert sha(stock)==plan['stock_weight_sha256']
p=r/a['checkpoint_path'];assert sha(p)==a['checkpoint_sha256'];trained=torch.load(p,map_location='cpu',weights_only=True)
assert trained['plan_sha256']==a['plan_sha256'] and trained['optimizer_updates']==1
before=torch.load(stock,map_location='cpu',weights_only=True);after=trained['state_dict']
changed=[k for k in before if k in after and isinstance(before[k],torch.Tensor) and before[k].is_floating_point() and before[k].shape==after[k].shape and not torch.equal(before[k],after[k])]
assert changed
result=dict(all_checks_passed=True,audit_script_sha256=sha(__file__),source_result_sha256=sha(run/'stage65_gradient_smoke.json'),
    checkpoint_sha256=sha(p),optimizer_updates=1,common_floating_tensors_changed=len(changed),changed_parameter_tensors_reported=a['changed_parameter_tensors'],
    training_image_identity_verified=True,frozen_evaluation_images_used_for_training=False,paper_baseline_ready=False)
(run/'stage65_gradient_smoke_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
