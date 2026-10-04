"""Stage 2: validation-only image ablations and training-only memorization controls.

No test split is opened. Overfit checkpoints are diagnostics, not production models.
AWGN 120 dB retains channel projections; kind=none bypasses them entirely.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from run import read_data, batch, decode
from lora import load_taichi
from real_maskgit import load_real_maskgit
from models import erase_packets
from pretrained_vq import sha256


def first_per_class(records):
    seen=set(); indices=[]
    for i,r in enumerate(records):
        if r[3] not in seen: seen.add(r[3]); indices.append(i)
    return indices


def image_variants(records):
    # Pair every image with the first image of the next class, preserving questions/targets.
    classes=sorted(set(r[3] for r in records))
    sources={c:next(r[0] for r in records if r[3]==c) for c in classes}
    return {'correct':records,
            'wrong_class':[(sources[classes[(classes.index(r[3])+1)%len(classes)]],*r[1:]) for r in records],
            'zero':[(torch.zeros_like(r[0]),*r[1:]) for r in records]}


@torch.inference_mode()
def qa_score(model,records,kind='awgn',snr=120.,seed=2040):
    model.eval(); predictions=[]; total=0.; count=0; first=[]
    # Independent, matched RNG streams for CE and generation in each variant.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(records),8):
            im,q,ans,_=batch(records[start:start+8],'cpu')
            logits=model(im,q,ans[:,:-1],snr,kind)
            total+=float(F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0,reduction='sum'))
            count+=int(ans[:,1:].ne(0).sum()); first.append(logits[:,0].cpu())
        torch.manual_seed(seed)
        for start in range(0,len(records),8):
            im,q,_,_=batch(records[start:start+8],'cpu')
            predictions.extend(decode(x) for x in model.generate(im,q,snr,kind,max_new_tokens=24).tolist())
    return {'ce':total/count,'exact_match':sum(p==r[2] for p,r in zip(predictions,records))/len(records),
            'predictions':predictions,'targets':[r[2] for r in records]},torch.cat(first)


def ablations(model,records,seeds):
    rows=[]
    for kind,snr in [('awgn',10.),('awgn',120.),('none',0.)]:
        for seed in seeds:
            original=None; orig_logits=None
            for variant,data in image_variants(records).items():
                score,logits=qa_score(model,data,kind,snr,seed)
                if original is None: original=score; orig_logits=logits
                rows.append({'channel':kind,'snr_db':snr,'seed':seed,'image_variant':variant,**score,
                    'prediction_change_fraction':sum(x!=y for x,y in zip(score['predictions'],original['predictions']))/len(records),
                    'first_token_logits_mean_abs_delta':float((logits-orig_logits).abs().mean())})
    return rows


@torch.inference_mode()
def token_score(model,ids,labels,seed=3001):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(seed); mask=erase_packets(ids,.3,4)
    logits=model(ids.masked_fill(mask,model.codes),labels)
    recovered=model.recover(ids.masked_fill(mask,model.codes),mask,labels)
    assert torch.equal(recovered[~mask],ids[~mask])
    return {'ce':float(F.cross_entropy(logits[mask],ids[mask])),
            'single_pass_accuracy':float((logits.argmax(-1)[mask]==ids[mask]).float().mean()),
            'iterative_accuracy':float((recovered[mask]==ids[mask]).float().mean()),
            'missing_tokens':int(mask.sum()),'seed':seed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/stage2_diagnostics')
    p.add_argument('--steps',type=int,default=800); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(2030)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists(): p.error('Use a new output directory')
    if a.steps<1: p.error('steps must be positive')
    root=Path('data/imagenette_pilot'); train=read_data(root/'train.jsonl'); val=read_data(root/'val.jsonl')
    tc=torch.load('runs/taichi_imagenette/best.pt',map_location='cpu',weights_only=True)
    mc=torch.load('runs/maskgit_imagenette/best.pt',map_location='cpu',weights_only=True)
    cache=torch.load(root/'token_cache/train.pt',weights_only=True)
    assert tc['dataset_identity']==sha256(root/'manifest.json')
    assert mc['identity']==cache['identity']
    idx=first_per_class(train); tiny=[train[i] for i in idx]
    config={'steps':a.steps,'threads':a.threads,'train_indices':idx,'seed':2030,
            'validation_images':len(val),'overfit_images':len(tiny),'test_split_opened':False,
            'dataset_identity':tc['dataset_identity'],'checkpoint_taichi_step':tc['step'],
            'checkpoint_maskgit_step':mc['step'],'overfit_lr':.001,'scope':'diagnostics, not benchmark or production weights'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    taichi=load_taichi(tc)
    before=ablations(taichi,val,[2040,2041])
    (out/'taichi_ablation_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    print('Validation image ablations saved',flush=True)
    im,q,ans,_=batch(tiny,'cpu'); im.requires_grad_(True)
    taichi.train(); loss=F.cross_entropy(taichi(im,q,ans[:,:-1],120.,'awgn').flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
    loss.backward()
    grads={'image_mean_abs_gradient':float(im.grad.abs().mean()),'modules':{}}
    for name in ['lr_patch','lr_vit','hr_cnn','ban','kan','encoder','channel_encoder','channel_decoder','decoder']:
        vals=[p.grad.square().sum() for p in getattr(taichi,name).parameters() if p.grad is not None]
        grads['modules'][name]=float(torch.stack(vals).sum().sqrt()) if vals else None
    taichi.zero_grad(set_to_none=True)
    (out/'gradient_diagnostics.json').write_text(json.dumps(grads,indent=2),encoding='utf-8')
    initial,_=qa_score(taichi,tiny)
    opt=torch.optim.AdamW(taichi.parameters(),lr=.001,weight_decay=0.)
    im=im.detach(); start=time.monotonic()
    with (out/'taichi_overfit_history.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for step in range(1,a.steps+1):
            taichi.train(); opt.zero_grad(set_to_none=True)
            logits=taichi(im,q,ans[:,:-1],120.,'awgn')
            loss=F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite TaiChi loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(taichi.parameters(),1.); opt.step()
            row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
            if step%100==0 or step==a.steps:
                row['train_diagnostic']=qa_score(taichi,tiny)[0]
                print('TaiChi',step,row['loss'],row['train_diagnostic']['exact_match'],flush=True)
            log.write(json.dumps(row)+'\n')
    torch.save({'model':'taichi','state':taichi.state_dict(),'optimizer':opt.state_dict(),'step':a.steps,
                'rng_cpu':torch.get_rng_state(),'dataset_identity':tc['dataset_identity'],'diagnostic_only':True,'config':config},out/'taichi_overfit.pt')
    after=ablations(taichi,tiny,[2040])
    val_after=qa_score(taichi,val)[0]
    taichi_result={'initial_train':initial,'after_train_ablations':after,'validation_after_high_snr':val_after}
    (out/'taichi_overfit_result.json').write_text(json.dumps(taichi_result,indent=2),encoding='utf-8')
    del taichi,opt
    maskgit=load_real_maskgit(mc); ids=cache['ids'][idx]; labels=cache['labels'][idx]
    initial_tokens=token_score(maskgit,ids,labels)
    opt=torch.optim.AdamW(maskgit.parameters(),lr=.001,weight_decay=0.)
    start=time.monotonic()
    with (out/'maskgit_overfit_history.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for step in range(1,a.steps+1):
            maskgit.train(); opt.zero_grad(set_to_none=True)
            mask=erase_packets(ids,.3,4)
            logits=maskgit(ids.masked_fill(mask,maskgit.codes),labels)
            loss=F.cross_entropy(logits[mask],ids[mask])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite MaskGIT loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(maskgit.parameters(),1.); opt.step()
            row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
            if step%100==0 or step==a.steps:
                row['fixed_train_diagnostic']=token_score(maskgit,ids,labels)
                print('MaskGIT',step,row['loss'],row['fixed_train_diagnostic'],flush=True)
            log.write(json.dumps(row)+'\n')
    torch.save({'model':'maskgit_real','config':maskgit.config,'state':maskgit.state_dict(),'optimizer':opt.state_dict(),
                'step':a.steps,'identity':mc['identity'],'diagnostic_only':True,'diagnostic_config':config,
                'rng_cpu':torch.get_rng_state()},out/'maskgit_overfit.pt')
    final_tokens=[token_score(maskgit,ids,labels,seed) for seed in [3001,3002,3003]]
    valcache=torch.load(root/'token_cache/val.pt',weights_only=True)
    valtoken=[]
    for i in range(0,len(valcache['ids']),10):
        valtoken.append(token_score(maskgit,valcache['ids'][i:i+10],valcache['labels'][i:i+10],4001+i))
    result={'initial_train':initial_tokens,'final_train_new_masks':final_tokens,
            'validation_after':valtoken,'validation_iterative_accuracy':sum(x['iterative_accuracy']*x['missing_tokens'] for x in valtoken)/sum(x['missing_tokens'] for x in valtoken)}
    (out/'maskgit_overfit_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'steps_per_model':a.steps,'test_split_opened':False}),encoding='utf-8')
    print('Completed stage 2 diagnostics',flush=True)


if __name__=='__main__': main()
