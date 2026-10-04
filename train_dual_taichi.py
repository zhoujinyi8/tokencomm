"""Train TaiChi with distinct frozen ViT-B/16 and ConvNeXt-Tiny cached features."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from pretrained_taichi_dual import DualBackboneTaiChi, load_dual_taichi
from run import encode, pad, decode


def get_batch(data, indices, variant='correct'):
    indices = torch.as_tensor(indices)
    vit = data['vit'][indices].float(); conv = data['conv'][indices].float()
    if variant == 'zero': vit = torch.zeros_like(vit); conv = torch.zeros_like(conv)
    if variant == 'wrong_class':
        donors = torch.tensor([int((data['labels'] == (int(c)+1)%10).nonzero()[0]) for c in data['labels'][indices]])
        vit = data['vit'][donors].float(); conv = data['conv'][donors].float()
    q = pad([encode(data['records'][int(i)]['question']) for i in indices], 'cpu')
    ans = pad([[1]+encode(data['records'][int(i)]['answer'])+[2] for i in indices], 'cpu')
    return (vit, conv), q, ans


@torch.inference_mode()
def evaluate(model, data, kind='awgn', snr=10., variant='correct', seed=8800):
    model.eval(); total=0.; count=0; pred=[]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0, len(data['labels']), 8):
            images,q,a=get_batch(data,range(start,min(start+8,len(data['labels']))),variant)
            logits=model(images,q,a[:,:-1],snr,kind)
            total += float(F.cross_entropy(logits.flatten(0,1),a[:,1:].reshape(-1),ignore_index=0,reduction='sum'))
            count += int(a[:,1:].ne(0).sum())
        torch.manual_seed(seed)
        for start in range(0, len(data['labels']), 8):
            images,q,_=get_batch(data,range(start,min(start+8,len(data['labels']))),variant)
            pred.extend(decode(x) for x in model.generate(images,q,snr,kind,max_new_tokens=24).tolist())
    targets=[r['answer'] for r in data['records']]
    return {'exact_match':sum(x==y for x,y in zip(pred,targets))/len(targets),'ce':total/count,
            'predictions':pred,'targets':targets,'channel':kind,'snr_db':snr,'variant':variant,'seed':seed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_stage7/dual_vision_features')
    p.add_argument('--out',default='runs/stage8_dual_taichi')
    p.add_argument('--init',default='runs/stage5_pretrained_taichi/best.pt')
    p.add_argument('--steps',type=int,default=1200); p.add_argument('--align-steps',type=int,default=200)
    a=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(8801)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): p.error('Use a new empty output directory')
    root=Path(a.data); train=torch.load(root/'train.pt',weights_only=True); val=torch.load(root/'val.pt',weights_only=True)
    if train['identity'] != val['identity'] or train['identity'].get('test_opened') is not False: raise ValueError('Feature identity/test lock mismatch')
    model=DualBackboneTaiChi(); model.initialize_shared(torch.load(a.init,map_location='cpu',weights_only=True))
    groups=[(train['labels']==c).nonzero().flatten() for c in range(10)]
    cfg={'steps':a.steps,'alignment_steps':a.align_steps,'lr':.0005,'batch_size':20,'seed':8801,
         'train_images':len(train['labels']),'validation_images':len(val['labels']),'test_accessed':False,
         'identity':train['identity'],'initialization':a.init,'pretrained_backbone_parameters_frozen':True,
         'trainable_parameters':sum(x.numel() for x in model.parameters()),
         'architecture':'distinct frozen ViT-B/16 + ConvNeXt-Tiny + adapters + BAN/KAN + language/channel modules',
         'paper_gap':'public ImageNet classification weights and tiny language module, not paper CLIP/Gemma/Qwen weights',
         'selection':'max validation exact match then min CE; AWGN10 fixed seed8800'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    initial=evaluate(model,val); (out/'initial_validation.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    best_key=(initial['exact_match'],-initial['ce']); best_step=0
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00005)
    def checkpoint(step): return {'model':'taichi_vit_b16_convnext_tiny_features','state':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'rng_cpu':torch.get_rng_state(),'step':step,'best_step':best_step,'feature_identity':train['identity'],'config':cfg}
    torch.save(checkpoint(0),out/'best.pt')
    for name,param in model.named_parameters(): param.requires_grad_(name.startswith(('vit_adapter.','conv_adapter.','ban.','kan.')))
    start=time.monotonic()
    with (out/'history.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for step in range(1,a.steps+1):
            if step==a.align_steps+1: model.requires_grad_(True)
            model.train(); inds=torch.cat([g[torch.randint(len(g),(2,))] for g in groups])
            images,q,ans=get_batch(train,inds); snr=120. if step<=a.align_steps else float(torch.rand(())*15)
            opt.zero_grad(set_to_none=True); logits=model(images,q,ans[:,:-1],snr,'awgn')
            loss=F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sch.step()
            row={'step':step,'loss':float(loss.detach()),'snr':snr,'seconds':time.monotonic()-start}
            if step%100==0 or step==a.steps:
                metric=evaluate(model,val); row['validation']={k:v for k,v in metric.items() if k not in ('predictions','targets')}
                key=(metric['exact_match'],-metric['ce'])
                if key>best_key: best_key=key; best_step=step; torch.save(checkpoint(step),out/'best.pt')
                torch.save(checkpoint(step),out/'last.pt'); print(row,flush=True)
            f.write(json.dumps(row)+'\n')
    best=load_dual_taichi(torch.load(out/'best.pt',weights_only=True)); results=[]
    for kind,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
        for variant in ['correct','wrong_class','zero']:
            for seed in [8800,8802]: results.append(evaluate(best,val,kind,snr,variant,seed))
    (out/'results.json').write_text(json.dumps({'best_step':best_step,'validation_ablations':results,'test_accessed':False},indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'best_step':best_step,'test_accessed':False}),encoding='utf-8')


if __name__=='__main__': main()
