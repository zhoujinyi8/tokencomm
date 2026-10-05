"""One held-out evaluation after validation-based checkpoint selection."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from real_maskgit import load_real_maskgit
from lora import load_taichi
from models import erase_packets
from run_pretrained import spatial_fill
from pretrained_vq import PretrainedVQGAN
from cache_real_tokens import load_image
from run import read_data,batch,decode


def save_image(x,path):
    Image.fromarray((x.cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).save(path)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_pilot')
    p.add_argument('--maskgit',default='runs/maskgit_imagenette/best.pt')
    p.add_argument('--taichi',default='runs/taichi_imagenette/best.pt')
    p.add_argument('--out',default='runs/real_evaluation')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--skip-images',action='store_true')
    a=p.parse_args(); torch.set_num_threads(a.threads)
    for checkpoint in [a.maskgit,a.taichi]:
        status=Path(checkpoint).parent/'status.json'
        if not status.exists() or not json.loads(status.read_text())['completed']:
            raise RuntimeError('Complete planned training before held-out evaluation: '+checkpoint)
    root=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    test=torch.load(root/'token_cache/test.pt',weights_only=True)
    train=torch.load(root/'token_cache/train.pt',weights_only=True)
    ck=torch.load(a.maskgit,map_location='cpu',weights_only=True)
    if ck['identity']!=test['identity']: raise ValueError('Checkpoint/tokenizer/dataset mismatch')
    model=load_real_maskgit(ck).eval()
    ids=test['ids']; labels=test['labels']; n=len(ids)
    histogram=torch.bincount(train['ids'].flatten(),minlength=1024).float()+1
    global_mode=int(histogram.argmax()); freq=(histogram/histogram.sum())
    rows=[]; selected=None
    for per in [.1,.3,.5]:
        torch.manual_seed(20261000+int(per*100))
        missing=erase_packets(ids,per); receiver=ids.masked_fill(missing,0)
        outputs={'spatial':torch.from_numpy(spatial_fill(receiver.numpy(),missing.numpy())),
                 'frequency':ids.masked_fill(missing,global_mode)}
        for cmi in [True,False]:
            recovered=[]
            for start in range(0,n,a.batch_size):
                lab=labels[start:start+a.batch_size] if cmi else torch.full_like(labels[start:start+a.batch_size],model.classes)
                recovered.append(model.recover(receiver[start:start+a.batch_size],missing[start:start+a.batch_size],lab))
            outputs['maskgit_cmi' if cmi else 'maskgit_no_cmi']=torch.cat(recovered)
        for method,pred in outputs.items():
            if not torch.equal(pred[~missing],ids[~missing]): raise AssertionError('Known token modified')
            row={'per':per,'method':method,'missing_tokens':int(missing.sum()),
                 'masked_token_accuracy':float((pred[missing]==ids[missing]).float().mean()),
                 'whole_sequence_token_accuracy':float((pred==ids).float().mean())}
            rows.append(row); print(row,flush=True)
        if per==.3: selected=(missing,outputs)
    # Teacher-forced single-pass CE is distinct from iterative recovery accuracy.
    mask=selected[0]; total_ce=0.; count=0
    for start in range(0,n,a.batch_size):
        x=ids[start:start+a.batch_size]; mm=mask[start:start+a.batch_size]
        logits=model(x.masked_fill(mm,1024),labels[start:start+a.batch_size])[mm]
        total_ce+=float(F.cross_entropy(logits,x[mm],reduction='sum')); count+=int(mm.sum())
    report={'scope':'Imagenette pilot held-out test, not ImageNet100 benchmark',
            'test_images':n,'maskgit_best_step':ck['step'],'maskgit_validation_ce':ck['best_val_ce'],
            'test_single_pass_ce':total_ce/count,'frequency_baseline_test_ce':float(-freq[ids[mask]].log().mean()),
            'token_results':rows,'class_names':test['class_names'],'checkpoint_paths':{'maskgit':a.maskgit,'taichi':a.taichi},
            'cmi_assumption':'Ground-truth class label is perfectly known at receiver; no class prediction implied'}
    del model
    tck=torch.load(a.taichi,map_location='cpu',weights_only=True)
    from pretrained_vq import sha256
    if tck['dataset_identity']!=sha256(root/'manifest.json'): raise ValueError('TaiChi dataset mismatch')
    taichi=load_taichi(tck).eval(); records=read_data(root/'test.jsonl')
    qa_results=[]
    for channel,snr in [('awgn',10.),('rayleigh',0.),('rayleigh',10.)]:
        torch.manual_seed(1100); predictions=[]
        for start in range(0,len(records),a.batch_size):
            im,q,_,_=batch(records[start:start+a.batch_size],'cpu')
            predictions.extend(decode(x) for x in taichi.generate(im,q,snr,channel,max_new_tokens=24).tolist())
        exact=sum(pred==r[2] for pred,r in zip(predictions,records))/len(records)
        qa_results.append({'channel':channel,'snr_db':snr,'exact_match':exact,
                           'examples':[{'target':r[2],'predicted':pred} for r,pred in zip(records,predictions)]})
        print('TaiChi',channel,snr,exact,flush=True)
    report['taichi']={'best_step':tck['step'],'best_validation_ce':tck['best_val_ce'],
                       'task':'10-class image classification expressed as free text QA',
                       'majority_class_baseline_accuracy':.1,'results':qa_results}
    del taichi
    if not a.skip_images:
        vq=PretrainedVQGAN(); image_metrics=[]; gallery=[]
        # Predetermined first held-out sample of each class, never quality-selected.
        indices=[int((labels==c).nonzero()[0]) for c in range(len(test['class_names']))]
        for i in indices:
            original=load_image(root/test['records'][i]['image'])
            methods={'clean_vqgan':ids[i:i+1],
                     'spatial':selected[1]['spatial'][i:i+1],
                     'maskgit_cmi':selected[1]['maskgit_cmi'][i:i+1],
                     'maskgit_no_cmi':selected[1]['maskgit_no_cmi'][i:i+1]}
            decoded={name:vq.decode(value)[0] for name,value in methods.items()}
            metric={'test_index':i,'label':int(labels[i]),'image':test['records'][i]['image'],
                    'psnr_db':{name:-10*math.log10(max(float(F.mse_loss(x,original)),1e-12)) for name,x in decoded.items()}}
            image_metrics.append(metric)
            strip=torch.cat([original,decoded['clean_vqgan'],decoded['spatial'],decoded['maskgit_cmi']],2)
            save_image(strip,out/f'comparison_class_{int(labels[i])}.png')
            if len(gallery)<3: gallery.append(strip)
            print('Image metrics',metric,flush=True)
        save_image(torch.cat(gallery,1),out/'comparison.png')
        report['image_evaluation']={'images':len(indices),'selection':'First test image of each class',
                                  'per':.3,'per_image':image_metrics,
                                  'mean_psnr_db':{name:sum(r['psnr_db'][name] for r in image_metrics)/len(image_metrics) for name in methods},
                                  'comparison_columns':['original','clean VQGAN','spatial fill','trained MaskGIT CMI']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved held-out evaluation',out,flush=True)


if __name__=='__main__': main()
