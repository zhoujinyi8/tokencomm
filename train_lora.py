"""Joint BAN/KAN/channel + LoRA training on the existing TaiChi baseline."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from lora import attach_lora,configure_joint_lora,load_taichi
from run import synthetic,read_data,batch,decode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='runs/taichi/checkpoint.pt')
    p.add_argument('--out',default='runs/taichi_lora')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--alpha',type=float,default=8.)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--channel',choices=['awgn','rayleigh'],default='rayleigh')
    p.add_argument('--snr-min',type=float,default=-5.)
    p.add_argument('--snr-max',type=float,default=15.)
    p.add_argument('--data'); p.add_argument('--eval-data')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1 or a.snr_min>a.snr_max: p.error('Invalid training settings')
    if bool(a.data)!=bool(a.eval_data): p.error('Provide both train and evaluation JSONL')
    torch.manual_seed(43); random.seed(43); torch.set_num_threads(4)
    ck=torch.load(a.base,map_location=a.device,weights_only=True)
    m=load_taichi(ck,a.device)
    if ck.get('lora'):
        config=ck['lora']
    else:
        targets=attach_lora(m,a.rank,a.alpha)
        config={'rank':a.rank,'alpha':a.alpha,'targets':targets}
    configure_joint_lora(m)
    frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
    train=read_data(a.data) if a.data else synthetic(96,42)
    val=read_data(a.eval_data) if a.eval_data else synthetic(18,20260926)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
    losses=[]
    for step in range(a.steps):
        m.train(); im,q,ans,_=batch(random.choices(train,k=a.batch_size),a.device)
        snr=random.uniform(a.snr_min,a.snr_max)
        logits=m(im,q,ans[:,:-1],snr,a.channel)
        loss=F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step(); scheduler.step()
        losses.append({'step':step,'loss':float(loss.detach()),'snr_db':snr})
        if step%20==0: print(losses[-1],flush=True)
    for name,p in m.named_parameters():
        if name in frozen and not torch.equal(p.detach().cpu(),frozen[name]): raise AssertionError('Frozen base changed: '+name)
    m.eval(); im,q,_,_=batch(val,a.device); results=[]
    with torch.inference_mode():
        for snr in [-5.,0.,5.,10.,20.]:
            torch.manual_seed(100)
            pred=[decode(x) for x in m.generate(im,q,snr,a.channel).tolist()]
            results.append({'snr_db':snr,'exact_match':sum(x==r[2] for x,r in zip(pred,val))/len(val)})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':'taichi','state':m.state_dict(),'lora':config},out/'checkpoint.pt')
    summary={'scope':'LoRA on tiny TaiChi language backbone, not Gemma/Qwen', 'args':vars(a),
             'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
             'total_parameters':sum(p.numel() for p in m.parameters()),
             'lora_parameters':sum(p.numel() for n,p in m.named_parameters() if '.parametrizations.' in n and n.endswith(('.A','.B'))),
             'frozen_parameters_unchanged':True,'lora':config,'evaluation':results,'loss_history':losses}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['loss_history','lora']},indent=2))


if __name__=='__main__': main()
