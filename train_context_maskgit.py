"""Multi-image/class MaskGIT without class side information; no test access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from real_maskgit import load_real_maskgit
from models import erase_packets
from run_pretrained import spatial_fill


@torch.inference_mode()
def evaluate(model,data,seed,controls=False):
    model.eval(); ids=data['ids']; labels=data['labels']
    with torch.random.fork_rng():
        torch.manual_seed(seed); missing=erase_packets(ids,.3,4)
    condition=torch.full_like(labels,model.classes)
    modes=['correct_context','wrong_context','no_context'] if controls else ['correct_context']
    rows=[]
    # Deterministic cross-class donor per image; targets and masks remain unchanged.
    donors=torch.tensor([int((labels==(int(c)+1)%10).nonzero()[0]) for c in labels])
    for mode in modes:
        outputs=[]; ce=0.
        for i in range(0,len(ids),8):
            target=ids[i:i+8]; mask=missing[i:i+8]
            context=ids[donors[i:i+8]] if mode=='wrong_context' else target
            recover_mask=torch.ones_like(mask) if mode=='no_context' else mask
            receiver=context.masked_fill(recover_mask,model.codes)
            logits=model(receiver,condition[i:i+8])
            ce+=float(F.cross_entropy(logits[mask],target[mask],reduction='sum'))
            pred=model.recover(receiver,recover_mask,condition[i:i+8])
            assert torch.equal(pred[~recover_mask],context[~recover_mask])
            outputs.append(pred)
        pred=torch.cat(outputs)
        rows.append({'mode':mode,'ce_on_common_targets':ce/int(missing.sum()),'correct_missing':int((pred[missing]==ids[missing]).sum()),
                     'missing_tokens':int(missing.sum()),'accuracy':float((pred[missing]==ids[missing]).float().mean())})
    if controls:
        spatial=torch.from_numpy(spatial_fill(ids.masked_fill(missing,0).numpy(),missing.numpy()))
        rows.append({'mode':'spatial','accuracy':float((spatial[missing]==ids[missing]).float().mean()),'missing_tokens':int(missing.sum())})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--per-class',type=int,default=4); p.add_argument('--out',default='runs/stage3_context_maskgit')
    p.add_argument('--data',default='data/imagenette_pilot/token_cache')
    p.add_argument('--eval-every',type=int,default=200)
    a=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(6300)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():p.error('Use a new output directory')
    root=Path(a.data)
    full=torch.load(root/'train.pt',weights_only=True); val=torch.load(root/'val.pt',weights_only=True)
    idx=torch.cat([(full['labels']==c).nonzero().flatten()[:a.per_class] for c in range(10)])
    if len(idx)!=10*a.per_class:p.error('Not enough samples')
    data={**full,'ids':full['ids'][idx],'labels':full['labels'][idx]}
    ck=torch.load('runs/maskgit_imagenette/best.pt',map_location='cpu',weights_only=True)
    if full['identity']!=val['identity']:raise ValueError('Train/validation cache identity mismatch')
    if full['identity']['tokenizer_sha256']!=ck['identity']['tokenizer_sha256']:
        raise ValueError('Initialization checkpoint uses a different tokenizer')
    model=load_real_maskgit(ck); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.0001)
    config={'args':vars(a),'train_indices':idx.tolist(),'training_images':len(idx),'validation_images':len(val['ids']),
            'condition':'always no CMI','seed':6300,'batch_size':8,'lr':.001,'test_accessed':False,'identity':full['identity'],
            'initialization':'runs/maskgit_imagenette/best.pt','initialization_dataset_identity':ck['identity']}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    before={'train':evaluate(model,data,6400,True),'val':evaluate(model,val,6401,True)}
    (out/'before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    best=before['val'][0]['ce_on_common_targets'];best_step=0
    def checkpoint(step):
        return {'model':'maskgit_real','config':model.config,'state':model.state_dict(),'optimizer':opt.state_dict(),
                'scheduler':scheduler.state_dict(),'rng_cpu':torch.get_rng_state(),'step':step,'identity':full['identity'],
                'class_names':full['class_names'],'run_config':config,'best_val_ce':best,'best_step':best_step,
                'scope':'multi-image/class no-CMI diagnostic'}
    torch.save(checkpoint(0),out/'best.pt'); start=time.monotonic()
    with (out/'history.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for step in range(1,a.steps+1):
            model.train(); inds=torch.randint(len(idx),(8,)); ids=data['ids'][inds]
            mask=erase_packets(ids,.3,4); labels=torch.full((len(ids),),model.classes,dtype=torch.long)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(ids.masked_fill(mask,model.codes),labels)[mask],ids[mask])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step()
            row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
            if step%a.eval_every==0 or step==a.steps:
                row['train']=evaluate(model,data,6400)[0];row['validation']=evaluate(model,val,6401)[0]
                if row['validation']['ce_on_common_targets']<best:
                    best=row['validation']['ce_on_common_targets'];best_step=step;torch.save(checkpoint(step),out/'best.pt')
                torch.save(checkpoint(step),out/'last.pt');print(row,flush=True)
            f.write(json.dumps(row)+'\n')
    result={'last_step':a.steps,'best_step':best_step,'last':{'train':evaluate(model,data,6500,True),'val':evaluate(model,val,6501,True)}}
    chosen=torch.load(out/'best.pt',weights_only=True);model=load_real_maskgit(chosen)
    result['best']={'train':evaluate(model,data,6500,True),'val':evaluate(model,val,6501,True)}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'best_step':best_step,'test_accessed':False}),encoding='utf-8')
    print('Completed no-CMI multi-image context experiment',flush=True)


if __name__=='__main__':main()
