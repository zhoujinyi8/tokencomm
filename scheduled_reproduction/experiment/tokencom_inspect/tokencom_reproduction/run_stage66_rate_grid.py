"""Resumable fixed grid; all arms start from the identical stock model and RNG seed."""
import argparse,json,math,os,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from PIL import Image
from torchvision import transforms
from cheng_runtime import activate,load_stock,sha
from cheng_stream_container import pack
p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=40);args=p.parse_args();assert 0<args.max_minutes<=45
r,_,_=activate();run=r/'runs';plan_path=run/'stage66_rate_grid_plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
for n,h in plan['evaluation_manifest_sha256'].items():assert sha(r/n)==h
for n,h in plan['historical_evidence_sha256'].items():assert sha(run/n)==h
for image in plan['images']+plan['calibration_images']:assert sha(r/image['path'])==image['sha256']
torch.set_num_threads(plan['torch_threads']);stock=r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar';assert sha(stock)==plan['stock_weight_sha256']
from compressai.zoo.image import model_architectures
from compressai.losses import RateDistortionLoss
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);deadline=time.monotonic()+args.max_minutes*60
directory=r/'pretrained/cheng_rate_grid66';directory.mkdir(exist_ok=True)
case_root=run/'stage66_rate_calibration';case_root.mkdir(exist_ok=True)
summary=run/'stage66_rate_grid.json'
if summary.exists():
    progress=json.loads(summary.read_text(encoding='utf-8'));assert progress['plan_sha256']==sha(plan_path)
else:progress=dict(plan_sha256=sha(plan_path),script_sha256=sha(__file__),arms=[],complete=False,paper_baseline_ready=False,full_paper_reproduction_complete=False)
def save_progress():
    progress['recorded_utc']=datetime.now(timezone.utc).isoformat();progress['complete']=len(progress['arms'])==len(plan['arms']) and all(a['complete'] for a in progress['arms'])
    tmp=summary.with_suffix('.tmp.json');tmp.write_text(json.dumps(progress,indent=2),encoding='utf-8');os.replace(tmp,summary)
for arm in plan['arms']:
    existing=next((v for v in progress['arms'] if v['arm_id']==arm['arm_id']),None)
    checkpoint=directory/(arm['arm_id']+'.pt')
    if existing and existing['complete']:
        assert sha(checkpoint)==existing['checkpoint_sha256'];continue
    if time.monotonic()>=deadline:break
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['plan_sha256']==sha(plan_path) and saved['worker_sha256']==sha(__file__) and saved['arm']==arm
        model=model_architectures['cheng2020-anchor'].from_state_dict(saved['state_dict']).train();step=saved['step'];history=saved['history']
    else:model=load_stock('cheng2020-anchor',stock).train();step=0;history=[];saved=None
    params=dict(model.named_parameters());main=[v for n,v in params.items() if v.requires_grad and not n.endswith('.quantiles')];aux=[v for n,v in params.items() if v.requires_grad and n.endswith('.quantiles')]
    optimizer=torch.optim.Adam(main,lr=plan['learning_rate']);aux_optimizer=torch.optim.Adam(aux,lr=plan['aux_learning_rate'])
    if saved:
        optimizer.load_state_dict(saved['optimizer']);aux_optimizer.load_state_dict(saved['aux_optimizer']);torch.set_rng_state(saved['torch_rng'])
    else:torch.manual_seed(plan['seed'])
    criterion=RateDistortionLoss(lmbda=arm['lmbda'],metric='mse')
    def save_checkpoint():
        temp=checkpoint.with_suffix('.partial.pt')
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),aux_optimizer=aux_optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            step=step,history=history,plan_sha256=sha(plan_path),worker_sha256=sha(__file__),arm=arm),temp);os.replace(temp,checkpoint)
    while step<plan['steps'] and time.monotonic()<deadline:
        image=plan['images'][step]
        with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
        optimizer.zero_grad(set_to_none=True);aux_optimizer.zero_grad(set_to_none=True)
        losses=criterion(model(x),x);assert all(torch.isfinite(v).all() for v in losses.values());losses['loss'].backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in main)
        norm=float(torch.nn.utils.clip_grad_norm_(main,plan['max_grad_norm']));assert math.isfinite(norm)
        optimizer.step();aux_loss=model.aux_loss();assert torch.isfinite(aux_loss).all();aux_loss.backward();aux_optimizer.step();step+=1
        history.append(dict(step=step,image_sha256=image['sha256'],loss=float(losses['loss'].detach()),estimated_bpp=float(losses['bpp_loss'].detach()),mse=float(losses['mse_loss'].detach()),aux_loss=float(aux_loss.detach())))
        if step%10==0:print(f'{arm["arm_id"]}: update={step}/100 estimated_bpp={history[-1]["estimated_bpp"]:.6f}',flush=True)
        if step%25==0:save_checkpoint()
    model.eval();model.update(force=True);save_checkpoint();cases=[]
    if step==plan['steps']:
        with torch.inference_mode():
            for image in plan['calibration_images']:
                with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
                encoded=model.compress(x);streams=[v[0] for v in encoded['strings']];shape=tuple(map(int,encoded['shape']))
                blob=pack('cheng2020-anchor',1,256,256,shape,streams);pixels=model.decompress(encoded['strings'],shape)['x_hat']
                mse=float((pixels-x).square().mean());assert math.isfinite(mse) and mse>0
                path=case_root/f'{arm["arm_id"]}_image{image["image_index"]:02d}.ch65';path.write_bytes(blob)
                import hashlib
                pixel_hash=hashlib.sha256(pixels.cpu().numpy().astype('<f4',copy=False).tobytes()).hexdigest()
                cases.append(dict(image_index=image['image_index'],image_sha256=image['sha256'],container_sha256=sha(path),container_path=str(path.relative_to(r)),
                    decoded_pixel_sha256=pixel_hash,entropy_payload_bpp=sum(len(v) for v in streams)*8/65536,total_stored_bpp=len(blob)*8/65536,psnr_decoded=-10*math.log10(mse)))
                print(f'{arm["arm_id"]}: calibration={image["image_index"]+1}/10 actual_bpp={cases[-1]["entropy_payload_bpp"]:.6f}',flush=True)
    result=dict(arm_id=arm['arm_id'],lmbda=arm['lmbda'],updates_completed=step,expected_updates=plan['steps'],complete=step==plan['steps'] and len(cases)==10,
        checkpoint_sha256=sha(checkpoint),checkpoint_path=str(checkpoint.relative_to(r)),history=history,calibration_cases=cases)
    if existing:progress['arms'].remove(existing)
    progress['arms'].append(result);save_progress();del model,optimizer,aux_optimizer,params,main,aux,saved
save_progress()
assert sha(stock)==plan['stock_weight_sha256']
print(json.dumps(dict(complete=progress['complete'],arms=[dict(arm_id=a['arm_id'],updates=a['updates_completed'],calibration_cases=len(a['calibration_cases'])) for a in progress['arms']]),indent=2))
