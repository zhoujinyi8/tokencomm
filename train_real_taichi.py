"""Real-image 10-class instruction finetuning of tiny TaiChi; not full VQA or CLIP/LLM."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from lora import load_taichi
from run import read_data,batch


@torch.inference_mode()
def validate(model,data,device,batch_size=8):
    model.eval(); loss=0.; count=0
    devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(900)
        for start in range(0,len(data),batch_size):
            im,q,ans,_=batch(data[start:start+batch_size],device)
            logits=model(im,q,ans[:,:-1],10.,'awgn')
            loss+=float(F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0,reduction='sum'))
            count+=int(ans[:,1:].ne(0).sum())
    return {'response_ce':loss/count,'response_tokens':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_pilot')
    p.add_argument('--base',default='runs/taichi/checkpoint.pt')
    p.add_argument('--out',default='runs/taichi_imagenette')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume'); p.add_argument('--eval-every',type=int,default=50)
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.eval_every)<1: p.error('Invalid training setting')
    torch.set_num_threads(a.threads); torch.manual_seed(20260928); random.seed(20260928)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and not a.resume: p.error('Choose a new output directory')
    train=read_data(Path(a.data)/'train.jsonl'); val=read_data(Path(a.data)/'val.jsonl')
    from pretrained_vq import sha256
    identity=sha256(Path(a.data)/'manifest.json')
    ck=torch.load(a.resume or a.base,map_location=a.device,weights_only=True)
    if ck.get('lora'): p.error('This full finetuning script expects an unparameterized base; use train_lora.py for LoRA')
    m=load_taichi(ck,a.device); m.set_stage('scratch')
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
    first=0; best=float('inf'); best_step=0
    if a.resume:
        if ck['planned_steps']!=a.steps or ck['dataset_identity']!=identity: p.error('Resume configuration mismatch')
        if out.resolve()!=Path(a.resume).resolve().parent: p.error('Resume into original output directory')
        for key in ['batch_size','lr','eval_every','device','threads']:
            if getattr(a,key)!=ck['args'][key]: p.error('Exact resume requires unchanged '+key)
        opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_cpu'].cpu()); random.setstate(ck['rng_python'])
        if a.device.startswith('cuda'): torch.cuda.set_rng_state_all(ck['rng_cuda'])
        first=ck['step']; best=ck['best_val_ce']; best_step=ck['best_step']
    initial=validate(m,val,a.device,a.batch_size)
    if not a.resume: (out/'initial_validation.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    (out/'run_config.json').write_text(json.dumps({'args':vars(a),'dataset_identity':identity,
        'training_images':len(train),'validation_images':len(val),'test_used_during_training':False,
        'scope':'Tiny TaiChi, image classification expressed as QA, 32px input, all parameters finetuned',
        'parameters':sum(p.numel() for p in m.parameters()),'torch':str(torch.__version__)},indent=2),encoding='utf-8')
    def checkpoint(step):
        return {'model':'taichi','state':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                'step':step,'planned_steps':a.steps,'best_val_ce':best,'best_step':best_step,
                'dataset_identity':identity,'rng_cpu':torch.get_rng_state(),'rng_python':random.getstate(),
                'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'args':vars(a)}
    start=time.monotonic(); last_log=start
    print('Initial validation:',initial,flush=True)
    with (out/'history.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        for step in range(first,a.steps):
            m.train(); im,q,ans,_=batch(random.choices(train,k=a.batch_size),a.device)
            snr=random.uniform(0,15)
            logits=m(im,q,ans[:,:-1],snr,'awgn')
            loss=F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            opt.step(); scheduler.step()
            row={'step':step+1,'train_loss':float(loss.detach()),'snr_db':snr,
                 'elapsed_seconds':time.monotonic()-start,'lr':scheduler.get_last_lr()[0]}
            if (step+1)%a.eval_every==0 or step+1==a.steps:
                metric=validate(m,val,a.device,a.batch_size); row['validation']=metric
                if metric['response_ce']<best:
                    best=metric['response_ce']; best_step=step+1; torch.save(checkpoint(step+1),out/'best.pt')
                torch.save(checkpoint(step+1),out/'last.pt')
                (out/'status.json').write_text(json.dumps({**row,'best_step':best_step,'best_val_ce':best,
                    'completed':step+1==a.steps},indent=2),encoding='utf-8')
                print(row,flush=True); last_log=time.monotonic()
            elif time.monotonic()-last_log>20:
                print(row,flush=True); last_log=time.monotonic()
            f.write(json.dumps(row)+'\n')
    print('Finished. Best validation step',best_step,best,flush=True)


if __name__=='__main__': main()
