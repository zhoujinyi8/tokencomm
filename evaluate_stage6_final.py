"""One-shot evaluation of frozen stage-5 models on the new 100-image holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights
from PIL import Image

from cache_pretrained_vision import load_backbone,features
from cache_real_tokens import load_image
from pretrained_vq import PretrainedVQGAN,WEIGHT_SHA256,sha256
from pretrained_taichi import load_feature_taichi
from real_maskgit import load_real_maskgit
from models import erase_packets
from run import encode,pad,decode
from run_pretrained import spatial_fill


def records(root):
    return [json.loads(x) for x in (root/'test.jsonl').read_text(encoding='utf-8').splitlines()]


@torch.inference_mode()
def build_test_cache(root,out):
    rows=records(root);backbone,digest=load_backbone();transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    lows=[];highs=[]
    for start in range(0,len(rows),8):
        images=[]
        for row in rows[start:start+8]:
            path=root/row['image']
            if sha256(path)!=row['sha256']:raise ValueError('Test image changed')
            with Image.open(path) as im:images.append(transform(im.convert('RGB')))
        low,high=features(backbone,torch.stack(images));lows.append(low);highs.append(high)
    vq=PretrainedVQGAN();tokens=[]
    for i,row in enumerate(rows):
        tokens.append(vq.encode(load_image(root/row['image'])[None])[0].cpu())
        if i%20==0:print('VQ test cache',i+1,'/',len(rows),flush=True)
    result={'records':rows,'labels':torch.tensor([r['label'] for r in rows]),'lr':torch.cat(lows),'hr':torch.cat(highs),
            'ids':torch.stack(tokens),'resnet_sha256':digest,'vq_sha256':WEIGHT_SHA256,'manifest_sha256':sha256(root/'manifest.json')}
    torch.save(result,out/'test_cache.pt');return result


def fbatch(data,start,end,variant):
    idx=torch.arange(start,end);low=data['lr'][idx];high=data['hr'][idx]
    if variant=='zero':low=torch.zeros_like(low);high=torch.zeros_like(high)
    elif variant=='wrong_class':
        donors=torch.tensor([int((data['labels']==(int(c)+1)%10).nonzero()[0]) for c in data['labels'][idx]])
        low=data['lr'][donors];high=data['hr'][donors]
    q=pad([encode(data['records'][int(i)]['question']) for i in idx],'cpu')
    a=pad([[1]+encode(data['records'][int(i)]['answer'])+[2] for i in idx],'cpu')
    return (low,high),q,a


@torch.inference_mode()
def eval_taichi(model,data,channel,snr,variant,seed):
    model.eval();pred=[];loss=0.;count=0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),10):
            images,q,a=fbatch(data,start,min(start+10,len(data['labels'])),variant)
            logits=model(images,q,a[:,:-1],snr,channel)
            loss+=float(F.cross_entropy(logits.flatten(0,1),a[:,1:].reshape(-1),ignore_index=0,reduction='sum'));count+=int(a[:,1:].ne(0).sum())
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),10):
            images,q,_=fbatch(data,start,min(start+10,len(data['labels'])),variant)
            pred.extend(decode(x) for x in model.generate(images,q,snr,channel,max_new_tokens=24).tolist())
    targets=[r['answer'] for r in data['records']];correct=sum(x==y for x,y in zip(pred,targets))
    return {'channel':channel,'snr_db':snr,'variant':variant,'seed':seed,'correct':correct,'images':len(targets),
            'accuracy':correct/len(targets),'response_ce':loss/count,'predictions':pred,'targets':targets}


@torch.inference_mode()
def eval_maskgit(model,data,seeds):
    model.eval();ids=data['ids'];labels=torch.full_like(data['labels'],model.classes);rows=[]
    for per in [.1,.3,.5]:
        for seed in seeds:
            with torch.random.fork_rng():torch.manual_seed(seed+int(per*100));missing=erase_packets(ids,per,4)
            pred=model.recover(ids.masked_fill(missing,model.codes),missing,labels)
            spatial=torch.from_numpy(spatial_fill(ids.masked_fill(missing,0).numpy(),missing.numpy()))
            rows.append({'per':per,'seed':seed,'missing_tokens':int(missing.sum()),
                         'maskgit_correct':int((pred[missing]==ids[missing]).sum()),
                         'maskgit_accuracy':float((pred[missing]==ids[missing]).float().mean()),
                         'spatial_correct':int((spatial[missing]==ids[missing]).sum()),
                         'spatial_accuracy':float((spatial[missing]==ids[missing]).float().mean())})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='runs/stage6_final_test');a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):p.error('Use a new empty output directory')
    root=Path('data/imagenette_stage5');tp=Path('runs/stage5_pretrained_taichi/best.pt');mp=Path('runs/stage5_context_maskgit/best.pt')
    for run in [tp.parent,mp.parent]:
        s=json.loads((run/'status.json').read_text());
        if not s['completed'] or s['test_accessed'] is not False:raise RuntimeError('Training run was not frozen')
    protocol={'written_before_test_read':True,'dataset_manifest_sha256':sha256(root/'manifest.json'),'test_images':100,
        'taichi_checkpoint':str(tp),'taichi_checkpoint_sha256':sha256(tp),'maskgit_checkpoint':str(mp),'maskgit_checkpoint_sha256':sha256(mp),
        'taichi_conditions':[{'channel':'awgn','snr_db':10.},{'channel':'awgn','snr_db':120.},{'channel':'rayleigh','snr_db':10.}],
        'feature_variants':['correct','wrong_class','zero'],'taichi_seeds':[9400,9401,9402,9403,9404],
        'maskgit_pers':[.1,.3,.5],'maskgit_seeds':[9500,9501,9502,9503,9504],
        'maskgit_condition':'always no CMI','selection_or_tuning_on_test':False,
        'repeats_note':'Five channel/mask draws reuse the same 100 images and are descriptive correlated repeats.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    data=build_test_cache(root,out)
    tck=torch.load(tp,map_location='cpu',weights_only=True);mck=torch.load(mp,map_location='cpu',weights_only=True)
    if tck['feature_identity']['weight_sha256']!=data['resnet_sha256'] or tck['feature_identity']['manifest_sha256']!=data['manifest_sha256']:
        raise ValueError('TaiChi feature/dataset identity mismatch')
    if mck['identity']['tokenizer_sha256']!=data['vq_sha256'] or mck['identity']['manifest_sha256']!=data['manifest_sha256']:
        raise ValueError('MaskGIT tokenizer/dataset identity mismatch')
    taichi=load_feature_taichi(tck);trows=[]
    for cond in protocol['taichi_conditions']:
        for variant in protocol['feature_variants']:
            for seed in protocol['taichi_seeds']:
                row=eval_taichi(taichi,data,cond['channel'],cond['snr_db'],variant,seed);trows.append(row)
                print('TaiChi',cond,variant,seed,row['accuracy'],flush=True)
    mrows=eval_maskgit(load_real_maskgit(mck),data,protocol['maskgit_seeds'])
    result={'scope':'one-shot evaluation on new 100-image holdout; no post-test selection','protocol':protocol,
            'taichi_step':tck['step'],'maskgit_step':mck['step'],'taichi':trows,'maskgit':mrows,
            'pretraining_overlap':'ResNet18 was pretrained on ImageNet; Imagenette is an ImageNet subset.'}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'test_opened':True,'models_modified_after_test':False},indent=2),encoding='utf-8')
    print('Stage-6 final test completed; frozen models unchanged.',flush=True)


if __name__=='__main__':main()
