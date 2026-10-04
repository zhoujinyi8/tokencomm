"""Scalable Qwen2.5-LoRA TaiChi training with gradient accumulation and resumable checkpoints."""
import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import torch

from qwen_dual_taichi import QwenDualTaiChi


def batch(data, indices, variant='correct'):
    idx=torch.as_tensor(indices); vit=data['vit'][idx].float(); conv=data['conv'][idx].float()
    if variant=='zero': vit.zero_(); conv.zero_()
    elif variant=='wrong_class':
        donors=torch.tensor([int((data['labels']==(int(c)+1)%10).nonzero()[0]) for c in data['labels'][idx]])
        vit=data['vit'][donors].float(); conv=data['conv'][donors].float()
    records=[data['records'][int(i)] for i in idx]
    return (vit,conv),[r['question'] for r in records],[r['answer'] for r in records]


def balanced_indices(labels, limit):
    n=min(limit,len(labels)); per,extra=divmod(n,10); selected=[]
    for cls in range(10):
        pool=(labels==cls).nonzero().flatten()
        selected.extend(pool[:per+(cls<extra)].tolist())
    if len(selected)!=n: raise ValueError('Insufficient examples for a ten-class balanced validation subset')
    return selected


@torch.inference_mode()
def evaluate(model,data,limit=64,seed=9900,variant='correct'):
    model.eval(); n=min(limit,len(data['labels'])); loss=0.; count=0; predictions=[]; targets=[]
    # Manifests are grouped by class, so taking records[:limit] would bias a
    # reduced validation run toward the first labels. Build a deterministic,
    # near-balanced subset instead (class counts differ by at most one).
    selected=balanced_indices(data['labels'],n)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,n,2):
            images,q,a=batch(data,selected[start:min(start+2,n)],variant)
            out=model(images,q,a,10.,'awgn'); loss+=float(out.loss)*len(a); count+=len(a)
            predictions.extend(model.generate_answers(images,q,10.,'awgn',max_new_tokens=12)); targets.extend(a)
    return {'loss':loss/count,'exact_match':sum(x.lower()==y.lower() for x,y in zip(predictions,targets))/count,
            'count':count,'variant':variant,'predictions':predictions,'targets':targets}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_stage7/dual_vision_features')
    p.add_argument('--llm',default=r'D:/新建文件夹 (2)/2026-09-21-14-31-36/taichi-tokecom/models/qwen2.5-0.5b-instruct')
    p.add_argument('--out',default='runs/stage9_qwen_dual_taichi'); p.add_argument('--steps',type=int,default=400)
    p.add_argument('--batch-size',type=int,default=2); p.add_argument('--accumulate',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=100); p.add_argument('--eval-n',type=int,default=64)
    p.add_argument('--lora-rank',type=int,default=8); p.add_argument('--resume'); p.add_argument('--reset-best',action='store_true')
    p.add_argument('--fusion-variant',choices=['ban_kan','ban_mlp','concat_mlp'],default='ban_kan')
    p.add_argument('--weights-only-resume',action='store_true',help='Load trainable weights and global step but start a fresh optimizer/schedule')
    p.add_argument('--threads',type=int,default=12,help='CPU intra-op threads (ignored by CUDA kernels)')
    a=p.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(9901)
    root=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    train=torch.load(root/'train.pt',weights_only=True); val=torch.load(root/'val.pt',weights_only=True)
    if train['identity']!=val['identity'] or train['identity'].get('test_opened') is not False: raise ValueError('Data/test-lock mismatch')
    model=QwenDualTaiChi(a.llm,lora_rank=a.lora_rank,fusion_variant=a.fusion_variant); summary=model.trainable_parameter_summary()
    opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=2e-4,weight_decay=.01)
    start_step=0; best=(float('inf'),0.); ck=None
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        named=dict(model.named_parameters())
        for name,value in ck['trainable_state'].items(): named[name].data.copy_(value)
        start_step=ck['step']; best=(float('inf'),0.) if (a.reset_best or a.weights_only_resume) else tuple(ck['best'])
    elif any(out.iterdir()): p.error('Use an empty output directory unless --resume is provided')
    if a.steps<=start_step: p.error('--steps must exceed the resumed global step')
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps-start_step,eta_min=2e-5)
    if ck is not None and not a.weights_only_resume:
        opt.load_state_dict(ck['optimizer']); sch.load_state_dict(ck['scheduler']); torch.set_rng_state(ck['rng'])
    cfg={'args':vars(a),'parameters':summary,'train_images':len(train['labels']),'validation_images':len(val['labels']),
         'identity':train['identity'],'test_accessed':False,'precision':'float32 CPU; CUDA autocast bfloat16 when available',
         'effective_batch_size':a.batch_size*a.accumulate,'selection':'min validation loss then max exact match'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    groups=[(train['labels']==c).nonzero().flatten() for c in range(10)]; device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    def save(step,name):
        trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        torch.save({'model':'qwen_dual_taichi','trainable_state':trainable,'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'rng':torch.get_rng_state(),'step':step,'best':best,'config':cfg},out/name)
    started=time.monotonic()
    with (out/'history.jsonl').open('a' if start_step else 'w',encoding='utf-8',buffering=1) as log:
        for step in range(start_step+1,a.steps+1):
            model.train(); opt.zero_grad(set_to_none=True); total=0.
            for micro in range(a.accumulate):
                cls=(step*a.accumulate+micro)%10; g=groups[cls]; inds=g[torch.randint(len(g),(a.batch_size,))]
                images,q,ans=batch(train,inds); images=tuple(x.to(device) for x in images)
                snr=120. if step<=50 else float(torch.rand(())*15)
                amp=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()
                with amp: loss=model(images,q,ans,snr,'awgn').loss/a.accumulate
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_((x for x in model.parameters() if x.requires_grad),1.); opt.step(); sch.step()
            row={'step':step,'loss':total,'seconds':time.monotonic()-started}
            if step%a.eval_every==0 or step==a.steps:
                metrics=evaluate(model,val,a.eval_n,9900); row['validation']={k:v for k,v in metrics.items() if k not in ('predictions','targets')}
                key=(metrics['loss'],-metrics['exact_match'])
                if key<best: best=key; save(step,'best.pt')
                save(step,'last.pt'); (out/'latest_validation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(row,flush=True)
            log.write(json.dumps(row)+'\n')
    chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=True); named=dict(model.named_parameters())
    for name,value in chosen['trainable_state'].items(): named[name].data.copy_(value.to(named[name].device))
    controls=[evaluate(model,val,a.eval_n,9902,v) for v in ('correct','wrong_class','zero')]
    (out/'results.json').write_text(json.dumps({'best_step':chosen['step'],'validation_controls':controls,'test_accessed':False},indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'best_step':chosen['step'],'test_accessed':False}),encoding='utf-8')


if __name__=='__main__': main()
