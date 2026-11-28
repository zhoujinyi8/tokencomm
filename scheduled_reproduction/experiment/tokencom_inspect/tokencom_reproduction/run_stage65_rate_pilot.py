"""Fixed 100-update proxy pilot with resumable optimizer/RNG state and post-budget calibration only."""
import argparse,json,math,os,time
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from cheng_runtime import activate,load_stock,sha
from cheng_stream_container import pack
p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=12);args=p.parse_args();assert 0<args.max_minutes<=20
r,source,_=activate();run=r/'runs';plan=json.loads((run/'stage65_rate_pilot_plan.json').read_text(encoding='utf-8'))
assert plan['worker_sha256']==sha(__file__) and plan['runtime_helper_sha256']==sha(r/'cheng_runtime.py')
assert plan['container_helper_sha256']==sha(r/'cheng_stream_container.py')
for name,h in plan['evaluation_manifest_sha256'].items():assert sha(r/name)==h
for image in plan['images']:assert sha(r/image['path'])==image['sha256']
torch.set_num_threads(plan['torch_threads']);stock=r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar';assert sha(stock)==plan['stock_weight_sha256']
target=r/'pretrained/cheng_rate_pilot';target.mkdir(exist_ok=True);checkpoint=target/'latest.pt'
from compressai.zoo.image import model_architectures
if checkpoint.exists():
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['plan_sha256']==sha(run/'stage65_rate_pilot_plan.json') and saved['worker_sha256']==sha(__file__)
    model=model_architectures['cheng2020-anchor'].from_state_dict(saved['state_dict']).train();step=saved['step'];history=saved['history']
else:model=load_stock('cheng2020-anchor',stock).train();step=0;history=[];saved=None
params=dict(model.named_parameters());main=[v for n,v in params.items() if v.requires_grad and not n.endswith('.quantiles')];aux=[v for n,v in params.items() if v.requires_grad and n.endswith('.quantiles')]
optimizer=torch.optim.Adam(main,lr=plan['learning_rate']);aux_optimizer=torch.optim.Adam(aux,lr=plan['aux_learning_rate'])
if saved:
    optimizer.load_state_dict(saved['optimizer']);aux_optimizer.load_state_dict(saved['aux_optimizer']);torch.set_rng_state(saved['torch_rng'])
else:torch.manual_seed(plan['seed'])
from compressai.losses import RateDistortionLoss
criterion=RateDistortionLoss(lmbda=plan['lmbda'],metric='mse');transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
deadline=time.monotonic()+args.max_minutes*60
def save_checkpoint():
    temp=checkpoint.with_suffix('.partial.pt')
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),aux_optimizer=aux_optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        step=step,history=history,plan_sha256=sha(run/'stage65_rate_pilot_plan.json'),worker_sha256=sha(__file__)),temp);os.replace(temp,checkpoint)
while step<plan['steps'] and time.monotonic()<deadline:
    image=plan['images'][step]
    with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
    optimizer.zero_grad(set_to_none=True);aux_optimizer.zero_grad(set_to_none=True)
    losses=criterion(model(x),x);assert all(torch.isfinite(v).all() for v in losses.values());losses['loss'].backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in main)
    norm=float(torch.nn.utils.clip_grad_norm_(main,plan['max_grad_norm']));assert math.isfinite(norm)
    optimizer.step();aux_loss=model.aux_loss();assert torch.isfinite(aux_loss).all();aux_loss.backward();aux_optimizer.step();step+=1
    history.append(dict(step=step,image_sha256=image['sha256'],loss=float(losses['loss'].detach()),estimated_bpp=float(losses['bpp_loss'].detach()),mse=float(losses['mse_loss'].detach()),aux_loss=float(aux_loss.detach())))
    if step%10==0:print(f'Proxy pilot update={step}/100 estimated_bpp={history[-1]["estimated_bpp"]:.6f}',flush=True)
    if step%25==0:save_checkpoint()
model.eval();model.update(force=True);save_checkpoint()
cases=[]
if step==plan['steps']:
    directory=run/'stage65_rate_calibration';directory.mkdir(exist_ok=True)
    with torch.inference_mode():
        for image in plan['calibration_images']:
            assert sha(r/image['path'])==image['sha256']
            with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
            encoded=model.compress(x);streams=[v[0] for v in encoded['strings']];shape=tuple(map(int,encoded['shape']))
            blob=pack('cheng2020-anchor',1,256,256,shape,streams);pixels=model.decompress(encoded['strings'],shape)['x_hat']
            mse=float((pixels-x).square().mean());assert math.isfinite(mse) and mse>0
            path=directory/f'image{image["image_index"]:02d}.ch65';path.write_bytes(blob)
            cases.append(dict(image_index=image['image_index'],image_sha256=image['sha256'],container_sha256=sha(path),entropy_payload_bpp=sum(len(v) for v in streams)*8/65536,
                total_stored_bpp=len(blob)*8/65536,psnr_decoded=-10*math.log10(mse)))
            print(f'Proxy calibration image={image["image_index"]+1}/10 actual_bpp={cases[-1]["entropy_payload_bpp"]:.6f}',flush=True)
assert sha(stock)==plan['stock_weight_sha256']
result=dict(all_checks_passed=True,script_sha256=sha(__file__),plan_sha256=sha(run/'stage65_rate_pilot_plan.json'),updates_completed=step,expected_updates=100,
    complete=step==100 and len(cases)==10,checkpoint_sha256=sha(checkpoint),checkpoint_path=str(checkpoint.relative_to(r)),history=history,calibration_cases=cases,
    mean_entropy_payload_bpp=sum(v['entropy_payload_bpp'] for v in cases)/10 if len(cases)==10 else None,
    calibration_is_development_only=True,frozen_evaluation_images_used_for_training=False,stock_weights_unchanged=True,paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage65_rate_pilot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('history','calibration_cases')},indent=2))
