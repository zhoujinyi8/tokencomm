import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cheng_stream_container import unpack
p=argparse.ArgumentParser();p.add_argument('--verify-codec',action='store_true');args=p.parse_args()
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage65_rate_pilot_plan.json');result=read('stage65_rate_pilot.json')
assert plan['worker_sha256']==result['script_sha256']==sha(r/'run_stage65_rate_pilot.py') and result['plan_sha256']==sha(run/'stage65_rate_pilot_plan.json')
assert plan['runtime_helper_sha256']==sha(r/'cheng_runtime.py') and plan['container_helper_sha256']==sha(r/'cheng_stream_container.py')
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
for v in plan['images']:
    assert sha(r/v['path'])==v['sha256'] and v['sha256'] not in forbidden and v['pixel_sha256'] not in pixels and v['source_path'] not in sources
assert len(plan['images'])==len({v['sha256'] for v in plan['images']})==100
assert all(sum(v['label']==c for v in plan['images'])==10 for c in range(10))
stock=r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar';assert sha(stock)==plan['stock_weight_sha256']
checkpoint=r/result['checkpoint_path'];assert checkpoint.resolve().is_relative_to((r/'pretrained/cheng_rate_pilot').resolve())
assert sha(checkpoint)==result['checkpoint_sha256'];state=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert state['plan_sha256']==result['plan_sha256'] and state['worker_sha256']==result['script_sha256']
assert state['step']==result['updates_completed']==len(result['history'])<=100 and state['history']==result['history']
for j,v in enumerate(result['history']):
    assert v['step']==j+1 and v['image_sha256']==plan['images'][j]['sha256']
    assert all(math.isfinite(v[k]) for k in ('loss','estimated_bpp','mse','aux_loss'))
steps=[int(v['step']) for v in state['optimizer']['state'].values()]
aux_steps=[int(v['step']) for v in state['aux_optimizer']['state'].values()]
assert steps and aux_steps and max(steps)==max(aux_steps)==state['step']
for value in state['state_dict'].values():
    if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
cases=result['calibration_cases'];assert not cases or state['step']==100
if cases:assert len(cases)==10 and {v['image_index'] for v in cases}==set(range(10))
fresh=[];model=None
if args.verify_codec and cases:
    from cheng_runtime import activate
    activate();torch.set_num_threads(plan['torch_threads'])
    from compressai.zoo.image import model_architectures
    model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
for v in cases:
    image=plan['calibration_images'][v['image_index']];assert sha(r/image['path'])==image['sha256']==v['image_sha256']
    p=run/'stage65_rate_calibration'/f'image{v["image_index"]:02d}.ch65';assert sha(p)==v['container_sha256']
    parsed=unpack(p.read_bytes());assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256
    assert sum(len(s) for s in parsed['streams'])*8/65536==v['entropy_payload_bpp'] and p.stat().st_size*8/65536==v['total_stored_bpp']
    if model is not None:
        with torch.inference_mode():decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
        assert decoded.shape==(1,3,256,256) and np.isfinite(decoded).all()
        with Image.open(r/image['path']) as opened:original=np.asarray(opened.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
        score=-10*math.log10(float(np.mean(np.square(decoded-original),dtype=np.float64)));assert abs(score-v['psnr_decoded'])<=1e-4
        fresh.append(dict(image_index=v['image_index'],psnr_difference=score-v['psnr_decoded'],encoder_called=False))
mean=sum(v['entropy_payload_bpp'] for v in cases)/10 if len(cases)==10 else None
audit=dict(all_checks_passed=True,audit_script_sha256=sha(__file__),source_result_sha256=sha(run/'stage65_rate_pilot.json'),plan_sha256=result['plan_sha256'],
    updates_completed=state['step'],expected_updates=100,calibration_cases=len(cases),fresh_decoder_checks=fresh,
    complete=state['step']==100 and len(fresh)==10,training_image_identities_verified=100,forbidden_evaluation_manifests_verified=len(plan['evaluation_manifest_sha256']),
    optimizer_step_range=[min(steps),max(steps)],aux_optimizer_step_range=[min(aux_steps),max(aux_steps)],
    mean_entropy_payload_bpp=mean,within_diagnostic_rate_tolerance=(abs(mean-plan['target_entropy_bpp'])<=plan['target_entropy_bpp']*plan['diagnostic_relative_tolerance']) if mean is not None else None,
    checkpoint_sha256=result['checkpoint_sha256'],stock_weights_unchanged=True,frozen_evaluation_images_used_for_training=False,
    paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage65_rate_pilot_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
