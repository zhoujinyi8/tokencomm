"""Train on cached real images. Model selection uses validation only; no test loading."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from real_maskgit import RealMaskGIT,load_real_maskgit


def fixed_mask(ids,seed=700):
    g=torch.Generator().manual_seed(seed)
    return (torch.rand(ids.shape,generator=g)<.3).to(ids.device)


@torch.inference_mode()
def validate(model,data,batch_size,device):
    model.eval(); total=0.; count=0; correct=0
    masks=fixed_mask(data['ids'])
    for start in range(0,len(data['ids']),batch_size):
        ids=data['ids'][start:start+batch_size].to(device)
        labels=data['labels'][start:start+batch_size].to(device)
        mask=masks[start:start+batch_size].to(device)
        logits=model(ids.masked_fill(mask,1024),labels)[mask]
        total+=float(F.cross_entropy(logits,ids[mask],reduction='sum'))
        count+=int(mask.sum()); correct+=int((logits.argmax(-1)==ids[mask]).sum())
    return {'ce':total/count,'masked_token_accuracy':correct/count,'masked_tokens':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_pilot/token_cache')
    p.add_argument('--out',default='runs/maskgit_imagenette')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--d',type=int,default=128); p.add_argument('--layers',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',help='Exact continuation of interrupted last.pt with same planned steps')
    p.add_argument('--init',help='Warm start model weights; start a new optimizer and schedule')
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1 or a.eval_every<1 or a.d%4 or a.layers<1: p.error('Invalid training settings')
    if a.resume and a.init: p.error('Choose --resume or --init')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and not a.resume: p.error('Output already contains a run; choose a new --out')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    train=torch.load(Path(a.data)/'train.pt',weights_only=True)
    val=torch.load(Path(a.data)/'val.pt',weights_only=True)
    if train['identity']!=val['identity']: raise ValueError('Dataset/tokenizer identity mismatch')
    ck=torch.load(a.resume or a.init,map_location=a.device,weights_only=True) if a.resume or a.init else None
    if ck and ck['identity']!=train['identity']: raise ValueError('Checkpoint dataset/tokenizer identity mismatch')
    model=load_real_maskgit(ck,a.device) if ck else RealMaskGIT(len(train['class_names']),a.d,a.layers).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
    first=0; best=float('inf'); best_step=0
    if a.resume:
        if ck['planned_steps']!=a.steps: p.error('For a longer new run use --init; exact resume needs same --steps')
        if out.resolve()!=Path(a.resume).resolve().parent: p.error('Resume into the original output directory')
        for key in ['batch_size','lr','eval_every','device','threads']:
            if getattr(a,key)!=ck['args'][key]: p.error('Exact resume requires unchanged '+key)
        opt.load_state_dict(ck['optimizer']); schedule.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if a.device.startswith('cuda'): torch.cuda.set_rng_state_all(ck['rng_cuda'])
        first=ck['step']; best=ck['best_val_ce']; best_step=ck['best_step']
    baseline=validate(model,val,a.batch_size,a.device)
    if not a.resume:
        (out/'initial_validation.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    start_time=time.monotonic(); last_log=time.monotonic()
    def checkpoint(step):
        return {'model':'maskgit_real','config':model.config,'state':model.state_dict(),
                'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'step':step,
                'planned_steps':a.steps,'best_val_ce':best,'best_step':best_step,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'identity':train['identity'],'class_names':train['class_names'],'args':vars(a)}
    (out/'run_config.json').write_text(json.dumps({'args':vars(a),'model':model.config,
        'train_images':len(train['ids']),'validation_images':len(val['ids']),
        'parameters':sum(x.numel() for x in model.parameters()),'torch':str(torch.__version__),
        'identity':train['identity'],'test_used_during_training':False},indent=2),encoding='utf-8')
    print('Initial validation:',baseline,flush=True)
    with (out/'history.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        for step in range(first,a.steps):
            model.train(); indices=torch.randint(len(train['ids']),(a.batch_size,))
            loss=model.training_loss(train['ids'][indices].to(a.device),train['labels'][indices].to(a.device))
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('Non-finite gradient')
            opt.step(); schedule.step()
            row={'step':step+1,'train_loss':float(loss.detach()),'lr':schedule.get_last_lr()[0],
                 'elapsed_seconds':time.monotonic()-start_time}
            if (step+1)%a.eval_every==0 or step+1==a.steps:
                result=validate(model,val,a.batch_size,a.device); row['validation']=result
                if result['ce']<best:
                    best=result['ce']; best_step=step+1
                    torch.save(checkpoint(step+1),out/'best.pt')
                torch.save(checkpoint(step+1),out/'last.pt')
                (out/'status.json').write_text(json.dumps({**row,'best_step':best_step,'best_val_ce':best,
                    'completed':step+1==a.steps},indent=2),encoding='utf-8')
                print(row,flush=True); last_log=time.monotonic()
            elif time.monotonic()-last_log>20:
                print(row,flush=True); last_log=time.monotonic()
            log.write(json.dumps(row)+'\n')
    print('Finished. Best validation checkpoint:',best_step,best,flush=True)


if __name__=='__main__': main()
