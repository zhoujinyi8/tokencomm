"""Resumable multi-SNR/multi-seed channel evaluation on the locked development split."""
import argparse
import json
from pathlib import Path

import torch

from qwen_dual_taichi import QwenDualTaiChi
from train_qwen_dual_taichi import balanced_indices, batch


@torch.inference_mode()
def evaluate(model,data,indices,snr,kind,seed):
    model.eval(); predictions=[]; targets=[]; labels=[]; total_loss=0.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(indices),2):
            chosen=indices[start:start+2]; images,q,a=batch(data,chosen)
            out=model(images,q,a,snr,kind); total_loss+=float(out.loss)*len(a)
            predictions.extend(model.generate_answers(images,q,snr,kind,max_new_tokens=12))
            targets.extend(a); labels.extend(int(data['labels'][i]) for i in chosen)
    correct=[p.lower()==t.lower() for p,t in zip(predictions,targets)]
    per_class={str(c):sum(ok for ok,y in zip(correct,labels) if y==c)/sum(y==c for y in labels) for c in range(10)}
    return {'kind':kind,'snr_db':snr,'seed':seed,'count':len(indices),'loss':total_loss/len(indices),
            'exact_match':sum(correct)/len(correct),'per_class_accuracy':per_class}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_stage10/dual_vision_features')
    p.add_argument('--checkpoint',default='runs/stage10_qwen_strong/best.pt')
    p.add_argument('--llm',default=r'D:/新建文件夹 (2)/2026-09-21-14-31-36/taichi-tokecom/models/qwen2.5-0.5b-instruct')
    p.add_argument('--out',default='runs/stage10_channel_sweep')
    p.add_argument('--snr',type=float,nargs='+',default=[0,5,10,20]); p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    p.add_argument('--limit',type=int,default=100); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=torch.load(Path(a.data)/'val.pt',weights_only=True)
    if data['identity'].get('test_opened') is not False: raise ValueError('Development/test lock mismatch')
    indices=balanced_indices(data['labels'],min(a.limit,len(data['labels'])))
    model=QwenDualTaiChi(a.llm,lora_rank=8); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    named=dict(model.named_parameters())
    for name,value in ck['trainable_state'].items(): named[name].data.copy_(value)
    path=out/'results.jsonl'; done={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line); done[(row['kind'],float(row['snr_db']),int(row['seed']))]=row
    with path.open('a',encoding='utf-8',buffering=1) as stream:
        for kind in ('awgn','rayleigh'):
            for snr in a.snr:
                for seed in a.seeds:
                    key=(kind,float(snr),seed)
                    if key in done: continue
                    row=evaluate(model,data,indices,float(snr),kind,seed); stream.write(json.dumps(row)+'\n'); print(row,flush=True)
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]
    summary=[]
    for kind in ('awgn','rayleigh'):
        for snr in a.snr:
            group=[r for r in rows if r['kind']==kind and float(r['snr_db'])==float(snr) and r['seed'] in a.seeds]
            values=torch.tensor([r['exact_match'] for r in group]); losses=torch.tensor([r['loss'] for r in group])
            summary.append({'kind':kind,'snr_db':snr,'seeds':len(group),'exact_match_mean':float(values.mean()),
                            'exact_match_std':float(values.std(unbiased=False)),'loss_mean':float(losses.mean())})
    result={'checkpoint_step':ck['step'],'validation_images':len(indices),'test_accessed':False,'summary':summary}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
