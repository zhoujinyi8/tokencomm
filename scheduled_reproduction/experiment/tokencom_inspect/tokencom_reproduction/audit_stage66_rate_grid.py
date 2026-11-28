"""Fresh-process, decoder-only audit of every fixed-grid checkpoint and calibration stream."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cheng_stream_container import unpack
from cheng_rate_selection import select_candidate
p=argparse.ArgumentParser();p.add_argument('--verify-codec',action='store_true');args=p.parse_args()
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_rate_grid_plan.json');result=read('stage66_rate_grid.json');plan_hash=sha(run/'stage66_rate_grid_plan.json')
assert result['plan_sha256']==plan_hash and result['script_sha256']==sha(r/'run_stage66_rate_grid.py')
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
for n,h in plan['historical_evidence_sha256'].items():assert sha(run/n)==h
assert sha(r/'pretrained/cheng_rate_pilot/latest.pt')==plan['prior_checkpoint_sha256']
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
assert len(plan['images'])==len({v['sha256'] for v in plan['images']})==100
assert {v['label'] for v in plan['images']}==set(range(10)) and all(sum(v['label']==c for v in plan['images'])==10 for c in range(10))
for v in plan['images']:
    assert sha(r/v['path'])==v['sha256'] and v['sha256'] not in forbidden and v['pixel_sha256'] not in pixels and v['source_path'] not in sources
calibration=plan['calibration_images'];assert len(calibration)==10 and {v['image_index'] for v in calibration}==set(range(10))
assert not {v['sha256'] for v in plan['images']}&{v['sha256'] for v in calibration}
assert sha(r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar')==plan['stock_weight_sha256']
if args.verify_codec:
    from cheng_runtime import activate
    activate();torch.set_num_threads(plan['torch_threads'])
    from compressai.zoo.image import model_architectures
arms=[]
assert len({a['arm_id'] for a in result['arms']})==len(result['arms'])
for arm in result['arms']:
    specification=next(v for v in plan['arms'] if v['arm_id']==arm['arm_id']);assert arm['lmbda']==specification['lmbda']
    checkpoint=r/arm['checkpoint_path'];assert checkpoint.resolve().is_relative_to((r/'pretrained/cheng_rate_grid66').resolve())
    assert sha(checkpoint)==arm['checkpoint_sha256'];state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['plan_sha256']==plan_hash and state['worker_sha256']==result['script_sha256'] and state['arm']==specification
    assert state['step']==arm['updates_completed']==len(arm['history'])<=plan['steps'] and state['history']==arm['history']
    for j,v in enumerate(arm['history']):
        assert v['step']==j+1 and v['image_sha256']==plan['images'][j]['sha256']
        assert all(math.isfinite(v[k]) for k in ('loss','estimated_bpp','mse','aux_loss'))
        expected=v['estimated_bpp']+specification['lmbda']*255**2*v['mse'];assert abs(expected-v['loss'])<1e-4
    steps=[int(v['step']) for v in state['optimizer']['state'].values()];aux_steps=[int(v['step']) for v in state['aux_optimizer']['state'].values()]
    assert steps and aux_steps and min(steps)==max(steps)==min(aux_steps)==max(aux_steps)==state['step']
    assert all(g['lr']==plan['learning_rate'] for g in state['optimizer']['param_groups'])
    assert all(g['lr']==plan['aux_learning_rate'] for g in state['aux_optimizer']['param_groups'])
    for value in state['state_dict'].values():
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
    cases=arm['calibration_cases'];assert not cases or state['step']==plan['steps']
    if cases:assert len(cases)==10 and {v['image_index'] for v in cases}==set(range(10))
    fresh=[];model=None
    if args.verify_codec and cases:
        model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
    for v in cases:
        image=next(c for c in calibration if c['image_index']==v['image_index']);assert sha(r/image['path'])==image['sha256']==v['image_sha256']
        stream_path=r/v['container_path'];assert stream_path.resolve().is_relative_to((run/'stage66_rate_calibration').resolve())
        assert sha(stream_path)==v['container_sha256'];parsed=unpack(stream_path.read_bytes())
        assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256
        assert sum(len(s) for s in parsed['streams'])*8/65536==v['entropy_payload_bpp'] and stream_path.stat().st_size*8/65536==v['total_stored_bpp']
        if model is not None:
            with torch.inference_mode():decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
            assert decoded.shape==(1,3,256,256) and np.isfinite(decoded).all()
            assert hashlib.sha256(decoded.astype('<f4',copy=False).tobytes()).hexdigest()==v['decoded_pixel_sha256']
            with Image.open(r/image['path']) as opened:original=np.asarray(opened.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
            score=-10*math.log10(float(np.mean(np.square(decoded-original),dtype=np.float64)));assert abs(score-v['psnr_decoded'])<=1e-4
            fresh.append(dict(image_index=v['image_index'],pixel_hash_equal=True,psnr_difference=score-v['psnr_decoded'],encoder_called=False))
            print(f'Audited {arm["arm_id"]} image={v["image_index"]+1}/10',flush=True)
    arms.append(dict(arm_id=arm['arm_id'],checkpoint_sha256=arm['checkpoint_sha256'],updates_completed=state['step'],calibration_cases=len(cases),fresh_decoder_checks=fresh,
                     optimizer_step_range=[min(steps),max(steps)],aux_optimizer_step_range=[min(aux_steps),max(aux_steps)]))
    del model,state
complete=len(arms)==len(plan['arms']) and all(a['updates_completed']==plan['steps'] and len(a['fresh_decoder_checks'])==10 for a in arms)
assert result['complete']==(len(result['arms'])==len(plan['arms']) and all(a['complete'] for a in result['arms']))
selection=select_candidate(result['arms'],[a['arm_id'] for a in plan['arms']],plan['target_entropy_bpp'],plan['diagnostic_relative_tolerance']) if complete else None
audit=dict(all_checks_passed=True,complete=complete,audit_script_sha256=sha(__file__),plan_sha256=plan_hash,source_result_sha256=sha(run/'stage66_rate_grid.json'),
    arms=arms,selection=selection,training_image_identities_verified=100,forbidden_evaluation_manifests_verified=len(plan['evaluation_manifest_sha256']),
    historical_artifacts_unchanged=len(plan['historical_evidence_sha256']),prior_checkpoint_unchanged=True,stock_weights_unchanged=True,
    frozen_evaluation_images_used_for_training=False,paper_baseline_ready=False,paper_dataset_identity_established=False,full_paper_reproduction_complete=False)
(run/'stage66_rate_grid_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
