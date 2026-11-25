"""Load both checkpoints and sweep channel settings on independent toy images."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from models import TaiChi, VQTokenizer, MaskGIT, erase_packets
from run import synthetic, batch, decode, save_grid
from lora import load_taichi


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--taichi',default='runs/taichi/checkpoint.pt')
    p.add_argument('--tokcom',default='runs/tokcom/checkpoint.pt')
    p.add_argument('--out',default='runs/evaluation')
    p.add_argument('--trials',type=int,default=3)
    a=p.parse_args()
    if a.trials<1: p.error('--trials must be positive')
    torch.set_num_threads(4)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=synthetic(18,20260926); im,q,_,labels=batch(data,'cpu')
    # A sample image makes the single-image entry point immediately usable.
    from PIL import Image
    Image.fromarray((im[0].permute(1,2,0).numpy()*255).astype('uint8')).save(out/'sample_red.png')
    ck=torch.load(a.taichi,map_location='cpu',weights_only=True)
    if ck['model']!='taichi': raise ValueError('Wrong TaiChi checkpoint')
    taichi=load_taichi(ck).eval()
    ck=torch.load(a.tokcom,map_location='cpu',weights_only=True)
    if ck['model']!='tokcom' or ck['classes']!=3: raise ValueError('Expected toy three-class TokCom checkpoint')
    vq=VQTokenizer().eval(); vq.load_state_dict(ck['vq'])
    maskgit=MaskGIT(classes=3).eval(); maskgit.load_state_dict(ck['state'])
    rows=[]
    for channel in ['awgn','rayleigh']:
        for snr in [-5,0,5,10,20]:
            accuracy=[]
            for trial in range(a.trials):
                torch.manual_seed(100+trial)
                pred=[decode(x) for x in taichi.generate(im,q,snr,channel).tolist()]
                accuracy.append(sum(x==r[2] for x,r in zip(pred,data))/len(data))
            row={'model':'taichi','channel':channel,'snr_db':snr,'exact_match':sum(accuracy)/len(accuracy)}
            rows.append(row); print(row,flush=True)
    ids=vq.encode(im); clean=vq.decode(ids)
    def psnr(x): return -10*math.log10(max(float(F.mse_loss(x,im)),1e-12))
    for per in [0.,.1,.3,.5,1.]:
        for cmi in [False,True]:
            quality=[]; correct=0; erased=0
            for trial in range(a.trials):
                torch.manual_seed(500+trial)
                missing=erase_packets(ids,per)
                recovered=maskgit.recover(ids,missing,labels if cmi else torch.full_like(labels,3))
                if not torch.equal(recovered[~missing],ids[~missing]): raise AssertionError('Received tokens modified')
                rec=vq.decode(recovered); quality.append(psnr(rec))
                erased+=int(missing.sum()); correct+=int((recovered[missing]==ids[missing]).sum())
                if per==.3 and cmi and trial==0: save_grid([im[0],clean[0],rec[0]],out/'comparison.png')
            row={'model':'tokcom','per':per,'cmi':cmi,'psnr_db':sum(quality)/len(quality),
                 'missing_token_accuracy':correct/erased if erased else None,
                 'clean_tokenizer_psnr_db':psnr(clean)}
            rows.append(row); print(row,flush=True)
    report={'scope':'Synthetic squares, 18 images, NOT paper benchmarks',
            'trials':a.trials,'torch':str(torch.__version__),'device':'cpu','results':rows}
    (out/'sweep.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'sweep.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


if __name__=='__main__': main()
