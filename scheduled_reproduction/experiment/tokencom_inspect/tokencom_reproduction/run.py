"""Train either small baseline, save checkpoint, run held-out inference."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from models import TaiChi, VQTokenizer, MaskGIT, erase_packets


def encode(text, limit=120):
    return [b+4 for b in text.encode('utf-8')[:limit]]


def decode(ids):
    values=[]
    for i in ids:
        if i==2: break
        if i>=4: values.append(i-4)
    return bytes(values).decode('utf-8',errors='replace')


def pad(rows,device):
    x=torch.zeros(len(rows),max(map(len,rows)),dtype=torch.long,device=device)
    for i,r in enumerate(rows): x[i,:len(r)]=torch.tensor(r,device=device)
    return x


def synthetic(n, seed):
    g=torch.Generator().manual_seed(seed); records=[]
    colors=['red','green','blue']
    for i in range(n):
        label=i%3
        im=torch.rand(3,32,32,generator=g)*.08
        y,x=torch.randint(2,12,(2,),generator=g).tolist()
        im[label,y:y+18,x:x+18]=.9
        records.append((im,'What color is the square?',colors[label],label))
    return records


def read_data(path):
    records=[]; p=Path(path)
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        with Image.open(p.parent/r['image']) as im:
            array=np.array(im.convert('RGB').resize((32,32)),dtype=np.float32)/255.
        label=int(r['label'])
        if label<0: raise ValueError('label must be nonnegative')
        records.append((torch.from_numpy(array).permute(2,0,1),r['question'],r['answer'],label))
    if not records: raise ValueError('Empty dataset')
    return records


def batch(records,device):
    im=torch.stack([r[0] for r in records]).to(device)
    q=pad([encode(r[1]) or [3] for r in records],device)
    a=pad([[1]+encode(r[2])+[2] for r in records],device)
    labels=torch.tensor([r[3] for r in records],device=device)
    return im,q,a,labels


def save_grid(images,path):
    arrays=[(x.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8') for x in images]
    Image.fromarray(np.concatenate(arrays,axis=1)).resize((len(arrays)*128,128)).save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['taichi','tokcom'])
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--vq-steps',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--data',help='Training JSONL; omit for synthetic squares')
    p.add_argument('--eval-data',help='Separate held-out JSONL required with --data')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--channel',choices=['none','awgn','rayleigh'],default='awgn')
    p.add_argument('--per',type=float,default=.3)
    p.add_argument('--stage',choices=['scratch','align','sft','joint'],default='scratch')
    p.add_argument('--resume',help='Load trusted checkpoint from this project; zero steps for inference')
    p.add_argument('--out',default='runs/demo')
    a=p.parse_args()
    if a.steps<0 or a.vq_steps<0 or a.batch_size<1 or not 0<=a.per<=1: p.error('Invalid training/packet parameters')
    if a.data and not a.eval_data: p.error('--data requires independent --eval-data')
    if a.model=='taichi' and a.stage!='scratch' and not a.resume: p.error('Non-scratch stages require --resume; freezing random backbones is not meaningful')
    if a.model=='taichi' and a.stage in ('align','sft') and a.channel!='none': p.error('align/sft require --channel none')
    if a.model=='taichi' and a.stage=='joint' and a.channel=='none': p.error('joint requires a noisy channel')
    random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    device=torch.device(a.device); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    train=read_data(a.data) if a.data else synthetic(96,42)
    test=read_data(a.eval_data) if a.eval_data else synthetic(18,314)
    ck=torch.load(a.resume,map_location=device,weights_only=True) if a.resume else None
    if ck and ck['model']!=a.model: raise ValueError('Checkpoint model mismatch')
    if ck and ck.get('lora'): p.error('Use train_lora.py or infer.py for LoRA checkpoints')
    history=[]
    def sample(): return batch(random.choices(train,k=a.batch_size),device)
    def update(loss,opt,model,step,phase):
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        history.append({'phase':phase,'step':step,'loss':float(loss.detach())})
        if step%20==0: print(phase,step,round(float(loss.detach()),5),flush=True)
    if a.model=='taichi':
        model=TaiChi().to(device)
        if ck: model.load_state_dict(ck['state'])
        model.set_stage(a.stage)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        for step in range(a.steps):
            im,q,ans,_=sample(); model.train()
            logits=model(im,q,ans[:,:-1],a.snr,a.channel)
            loss=F.cross_entropy(logits.reshape(-1,260),ans[:,1:].reshape(-1),ignore_index=0)
            update(loss,opt,model,step,'taichi')
        model.eval(); im,q,ans,_=batch(test,device)
        predicted=[decode(r) for r in model.generate(im,q,a.snr,a.channel).tolist()]
        metrics={'exact_match':sum(x==r[2] for x,r in zip(predicted,test))/len(test),
                 'examples':[{'question':r[1],'target':r[2],'predicted':x} for x,r in zip(predicted,test)]}
        torch.save({'model':'taichi','state':model.state_dict()},out/'checkpoint.pt')
    else:
        classes=max(r[3] for r in train)+1
        if any(r[3]>=classes for r in test): raise ValueError('Evaluation contains unknown class IDs')
        if ck and classes!=ck['classes']: raise ValueError('Dataset class count differs from checkpoint')
        vq=VQTokenizer().to(device); model=MaskGIT(classes=classes).to(device)
        if ck: vq.load_state_dict(ck['vq']); model.load_state_dict(ck['state'])
        opt=torch.optim.AdamW(vq.parameters(),lr=2e-3)
        for step in range(a.vq_steps):
            im,_,_,_=sample(); vq.train(); _,_,loss=vq(im)
            update(loss,opt,vq,step,'vq')
        vq.eval()
        for par in vq.parameters(): par.requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for step in range(a.steps):
            im,_,_,labels=sample(); ids=vq.encode(im); model.train()
            labels=labels.masked_fill(torch.rand_like(labels,dtype=torch.float)<.15,classes)
            update(model.masked_loss(ids,labels),opt,model,step,'maskgit')
        model.eval(); im,_,_,labels=batch(test,device); ids=vq.encode(im)
        torch.manual_seed(2026); missing=erase_packets(ids,a.per)
        recovered=model.recover(ids,missing,labels)
        without=model.recover(ids,missing,torch.full_like(labels,classes))
        with torch.no_grad(): clean=vq.decode(ids); recon=vq.decode(recovered)
        mse=float(F.mse_loss(recon,im))
        metrics={'per_requested':a.per,'token_erasure_fraction':float(missing.float().mean()),
                 'missing_token_accuracy':float((recovered[missing]==ids[missing]).float().mean()) if missing.any() else None,
                 'missing_token_accuracy_no_cmi':float((without[missing]==ids[missing]).float().mean()) if missing.any() else None,
                 'reconstruction_mse':mse,'psnr_db':-10*math.log10(max(mse,1e-12)),
                 'source_payload_bpp':ids.shape[1]*6/(32*32),
                 'bpp_note':'Excludes CMI, headers, CRC and FEC; toy codebook is 64 entries.'}
        save_grid([im[0],clean[0],recon[0]],out/'reconstruction.png')
        torch.save({'model':'tokcom','classes':classes,'vq':vq.state_dict(),'state':model.state_dict()},out/'checkpoint.pt')
    report={'implementation':'small baseline, NOT paper benchmark reproduction','args':vars(a),
            'torch':torch.__version__,'device':str(device),'metrics':metrics,'loss_history':history}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Saved:',out.resolve())


if __name__=='__main__': main()
