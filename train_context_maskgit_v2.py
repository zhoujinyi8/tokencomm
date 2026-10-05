"""Train ContextMaskGITv2 on an unused split; no class condition and no test access."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from context_maskgit_v2 import ContextMaskGITv2,load_context_maskgit_v2
from models import erase_packets
from run_pretrained import spatial_fill

@torch.inference_mode()
def score(model,data,seed,controls=False,batch_size=32):
    model.eval();ids=data['ids'];rows=[]
    with torch.random.fork_rng():torch.manual_seed(seed);mask=erase_packets(ids,.3,4)
    donors=torch.tensor([int((data['labels']==(int(c)+1)%10).nonzero()[0]) for c in data['labels']])
    for mode in (['correct','wrong','none'] if controls else ['correct']):
        source=ids[donors] if mode=='wrong' else ids;active=torch.ones_like(mask) if mode=='none' else mask
        pred=[];total=0.
        for i in range(0,len(ids),batch_size):
            x=source[i:i+batch_size].masked_fill(active[i:i+batch_size],model.codes);logits=model(x)
            total+=float(F.cross_entropy(logits[mask[i:i+batch_size]],ids[i:i+batch_size][mask[i:i+batch_size]],reduction='sum'))
            pred.append(model.recover(x,active[i:i+batch_size]))
        pred=torch.cat(pred);rows.append({'mode':mode,'ce':total/int(mask.sum()),'accuracy':float((pred[mask]==ids[mask]).float().mean()),'missing_tokens':int(mask.sum())})
    if controls:
        spatial=torch.from_numpy(spatial_fill(ids.masked_fill(mask,0).numpy(),mask.numpy()))
        rows.append({'mode':'spatial','accuracy':float((spatial[mask]==ids[mask]).float().mean()),'missing_tokens':int(mask.sum())})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--d',type=int,default=128);p.add_argument('--layers',type=int,default=4);p.add_argument('--local-blocks',type=int,default=2);p.add_argument('--eval-every',type=int,default=400);p.add_argument('--eval-batch-size',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(9700)
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):p.error('Use a new output directory')
    train=torch.load(root/'train.pt',weights_only=True);val=torch.load(root/'val.pt',weights_only=True)
    if train['identity']!=val['identity']:raise ValueError('Token cache mismatch')
    model=ContextMaskGITv2(a.d,a.layers,a.local_blocks);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=5e-5)
    cfg={'args':vars(a),'identity':train['identity'],'training_images':len(train['ids']),'validation_images':len(val['ids']),'test_accessed':False,'condition':'none','seed':9700,'parameters':sum(x.numel() for x in model.parameters())}
    score_kwargs={'batch_size':a.eval_batch_size}
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');before={'train':score(model,train,9710,True,**score_kwargs),'val':score(model,val,9711,True,**score_kwargs)};(out/'before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    best=before['val'][0]['ce'];beststep=0
    def ck(step):return {'model':'context_maskgit_v2','config':model.config,'state':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'step':step,'best_step':beststep,'best_val_ce':best,'identity':train['identity'],'run_config':cfg}
    torch.save(ck(0),out/'best.pt');start=time.monotonic()
    with (out/'history.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for step in range(1,a.steps+1):
            inds=torch.randint(len(train['ids']),(8,));ids=train['ids'][inds];loss,_=model.loss(ids)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sch.step();row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
            if step%a.eval_every==0 or step==a.steps:
                row['validation']=score(model,val,9711,**score_kwargs)[0]
                if row['validation']['ce']<best:best=row['validation']['ce'];beststep=step;torch.save(ck(step),out/'best.pt')
                torch.save(ck(step),out/'last.pt');print(row,flush=True)
            f.write(json.dumps(row)+'\n')
    last={'train':score(model,train,9720,True,**score_kwargs),'val':score(model,val,9721,True,**score_kwargs)};best_model=load_context_maskgit_v2(torch.load(out/'best.pt',weights_only=True));chosen={'train':score(best_model,train,9720,True,**score_kwargs),'val':score(best_model,val,9721,True,**score_kwargs)}
    (out/'results.json').write_text(json.dumps({'best_step':beststep,'last':last,'best':chosen},indent=2),encoding='utf-8');(out/'status.json').write_text(json.dumps({'completed':True,'test_accessed':False,'best_step':beststep}),encoding='utf-8')
if __name__=='__main__':main()
