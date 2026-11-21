"""One-shot final evaluation of frozen stage-3 checkpoints on the untouched test split.

This script does not train, select a checkpoint, or overwrite an existing result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights
from PIL import Image

from cache_pretrained_vision import load_backbone, features
from pretrained_taichi import load_feature_taichi
from real_maskgit import load_real_maskgit
from models import erase_packets
from run import encode, pad, decode
from run_pretrained import spatial_fill
from pretrained_vq import sha256


ROOT = Path(__file__).resolve().parent


def state_hash(checkpoint):
    """Hash exact checkpoint bytes so the evaluated model is unambiguous."""
    return sha256(checkpoint)


def load_test_features(data_root, out):
    records=[json.loads(x) for x in (data_root/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    backbone,digest=load_backbone(); transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    lows=[]; highs=[]
    with torch.inference_mode():
        for start in range(0,len(records),8):
            batch=[]
            for row in records[start:start+8]:
                with Image.open(data_root/row['image']) as image:
                    batch.append(transform(image.convert('RGB')))
            low,high=features(backbone,torch.stack(batch));lows.append(low);highs.append(high)
    result={'lr':torch.cat(lows),'hr':torch.cat(highs),'labels':torch.tensor([r['label'] for r in records]),
            'records':records,'weight_sha256':digest,'manifest_sha256':sha256(data_root/'manifest.json')}
    torch.save(result,out/'test_features.pt')
    return result


def feature_batch(data, start, end, variant='correct'):
    indices=torch.arange(start,end);low=data['lr'][indices];high=data['hr'][indices]
    if variant=='zero':low=torch.zeros_like(low);high=torch.zeros_like(high)
    elif variant=='wrong_class':
        donors=torch.tensor([int((data['labels']==(int(c)+1)%10).nonzero()[0]) for c in data['labels'][indices]])
        low=data['lr'][donors];high=data['hr'][donors]
    q=pad([encode(data['records'][int(i)]['question']) for i in indices],'cpu')
    a=pad([[1]+encode(data['records'][int(i)]['answer'])+[2] for i in indices],'cpu')
    return (low,high),q,a


@torch.inference_mode()
def evaluate_taichi(model,data,channel,snr,seed,variant='correct'):
    model.eval();predictions=[];loss=0.;count=0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),8):
            images,q,a=feature_batch(data,start,min(start+8,len(data['labels'])),variant)
            logits=model(images,q,a[:,:-1],snr,channel)
            loss+=float(F.cross_entropy(logits.flatten(0,1),a[:,1:].reshape(-1),ignore_index=0,reduction='sum'))
            count+=int(a[:,1:].ne(0).sum())
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),8):
            images,q,_=feature_batch(data,start,min(start+8,len(data['labels'])),variant)
            predictions.extend(decode(x) for x in model.generate(images,q,snr,channel,max_new_tokens=24).tolist())
    targets=[r['answer'] for r in data['records']]
    return {'channel':channel,'snr_db':snr,'seed':seed,'variant':variant,'correct':sum(p==t for p,t in zip(predictions,targets)),
            'images':len(targets),'accuracy':sum(p==t for p,t in zip(predictions,targets))/len(targets),
            'response_ce':loss/count,'predictions':predictions,'targets':targets}


@torch.inference_mode()
def evaluate_maskgit(model,data,seeds,label_mode='no_cmi'):
    model.eval();rows=[];ids=data['ids'];labels=data['labels']
    condition=torch.full_like(labels,model.classes) if label_mode=='no_cmi' else labels
    for per in [.1,.3,.5]:
        for seed in seeds:
            with torch.random.fork_rng():torch.manual_seed(seed+int(per*100));missing=erase_packets(ids,per,4)
            receiver=ids.masked_fill(missing,model.codes);pred=model.recover(receiver,missing,condition)
            spatial=torch.from_numpy(spatial_fill(ids.masked_fill(missing,0).numpy(),missing.numpy()))
            assert torch.equal(pred[~missing],ids[~missing])
            rows.append({'per':per,'seed':seed,'label_mode':label_mode,'missing_tokens':int(missing.sum()),
                         'maskgit_correct':int((pred[missing]==ids[missing]).sum()),
                         'maskgit_accuracy':float((pred[missing]==ids[missing]).float().mean()),
                         'spatial_correct':int((spatial[missing]==ids[missing]).sum()),
                         'spatial_accuracy':float((spatial[missing]==ids[missing]).float().mean())})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/stage4_final_test');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):p.error('Final-test output must be a new empty directory')
    data_root=Path('data/imagenette_pilot')
    taichi_path=Path('runs/stage3_pretrained_taichi_v2/best.pt')
    maskgit_selected_path=Path('runs/stage3_context_maskgit/best.pt')
    maskgit_last_path=Path('runs/stage3_context_maskgit/last.pt')
    for run in [taichi_path.parent,maskgit_selected_path.parent]:
        status=json.loads((run/'status.json').read_text(encoding='utf-8'))
        if not status['completed'] or status.get('test_accessed') is not False:raise RuntimeError('Training run is not frozen before test')
    protocol={'created_before_results':True,'test_images':40,'taichi_checkpoint':str(taichi_path),
              'maskgit_selected_checkpoint':str(maskgit_selected_path),'maskgit_diagnostic_last':str(maskgit_last_path),
              'taichi_conditions':[{'channel':'awgn','snr_db':10.},{'channel':'awgn','snr_db':120.},{'channel':'rayleigh','snr_db':10.}],
              'taichi_variants':['correct','wrong_class','zero'],'taichi_seeds':[8400,8401,8402,8403,8404],
              'maskgit_pers':[.1,.3,.5],'maskgit_seeds':[8500,8501,8502,8503,8504],
              'maskgit_selected_label_mode':'no_cmi','selection_or_tuning_on_test':False,
              'interpretation':'Repeated channel/mask draws on the same 40 images are correlated descriptive repeats, not 200 independent images.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    features_test=load_test_features(data_root,out)
    tck=torch.load(taichi_path,map_location='cpu',weights_only=True)
    if tck['feature_identity']['weight_sha256']!=features_test['weight_sha256']:raise ValueError('Backbone identity mismatch')
    if tck['feature_identity']['manifest_sha256']!=features_test['manifest_sha256']:raise ValueError('Dataset identity mismatch')
    taichi=load_feature_taichi(tck)
    taichi_rows=[]
    for condition in protocol['taichi_conditions']:
        for variant in protocol['taichi_variants']:
            for seed in protocol['taichi_seeds']:
                taichi_rows.append(evaluate_taichi(taichi,features_test,condition['channel'],condition['snr_db'],seed,variant))
                print('TaiChi',condition,variant,seed,taichi_rows[-1]['accuracy'],flush=True)
    token_test=torch.load(data_root/'token_cache/test.pt',weights_only=True)
    selected_ck=torch.load(maskgit_selected_path,map_location='cpu',weights_only=True)
    last_ck=torch.load(maskgit_last_path,map_location='cpu',weights_only=True)
    if selected_ck['identity']!=token_test['identity'] or last_ck['identity']!=token_test['identity']:raise ValueError('Token cache identity mismatch')
    selected_rows=evaluate_maskgit(load_real_maskgit(selected_ck),token_test,protocol['maskgit_seeds'])
    last_rows=evaluate_maskgit(load_real_maskgit(last_ck),token_test,protocol['maskgit_seeds'])
    result={'scope':'one-shot held-out stage-4 final test; no tuning after observation','protocol':protocol,
            'checkpoint_sha256':{'taichi':state_hash(taichi_path),'maskgit_selected':state_hash(maskgit_selected_path),'maskgit_last_diagnostic':state_hash(maskgit_last_path)},
            'taichi_best_step':tck['step'],'maskgit_selected_step':selected_ck['step'],'maskgit_last_step':last_ck['step'],
            'taichi':taichi_rows,'maskgit_selected':selected_rows,'maskgit_last_diagnostic':last_rows,
            'pretraining_overlap':'ResNet18 is ImageNet pretrained and Imagenette is an ImageNet subset; this is not unseen-pretraining evaluation.'}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'test_opened':True,'models_modified_after_test':False},indent=2),encoding='utf-8')
    print('Final test saved. Do not tune these frozen runs using this result.',flush=True)


if __name__=='__main__':main()
