"""Single-image inference from a project checkpoint; no training required."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import TaiChi, VQTokenizer, MaskGIT, erase_packets
from run import encode, decode, save_grid
from lora import load_taichi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--question',default='What color is the square?')
    p.add_argument('--label',type=int,help='Reliable receiver class ID for TokCom; omit to disable CMI')
    p.add_argument('--channel',choices=['none','awgn','rayleigh'],default='awgn')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--per',type=float,default=.3)
    p.add_argument('--link',choices=['erasure','digital'],default='erasure',help='TokCom: manual PER or actual bit-level link')
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='runs/inference')
    a=p.parse_args()
    if not 0<=a.per<=1: p.error('PER must be in [0,1]')
    if a.link=='digital' and a.channel=='none': p.error('Digital link requires --channel awgn or rayleigh')
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    with Image.open(a.image) as image:
        ar=np.asarray(image.convert('RGB').resize((32,32)),dtype=np.float32)/255.
    im=torch.from_numpy(ar).permute(2,0,1).unsqueeze(0).to(a.device)
    result={'model':ck['model'],'image':str(Path(a.image).resolve()),'seed':a.seed,
            'scope':'Tiny model; provided demo checkpoints learned synthetic colored squares only.'}
    with torch.inference_mode():
        if ck['model']=='taichi':
            m=load_taichi(ck,a.device).eval()
            q=torch.tensor([encode(a.question) or [3]],device=a.device)
            result.update(question=a.question,channel=a.channel,snr_db=a.snr,
                          answer=decode(m.generate(im,q,a.snr,a.channel)[0].tolist()))
        elif ck['model']=='tokcom':
            if a.label is not None and not 0<=a.label<ck['classes']: p.error('Invalid class label for checkpoint')
            vq=VQTokenizer().to(a.device).eval(); vq.load_state_dict(ck['vq'])
            m=MaskGIT(classes=ck['classes']).to(a.device).eval(); m.load_state_dict(ck['state'])
            ids=vq.encode(im)
            if a.link=='digital':
                from digital_link import transmit_tokens
                received,mask,stats=transmit_tokens(ids.cpu().numpy(),64,a.snr,a.channel,a.seed)
                receiver_ids=torch.tensor(received,device=a.device)
                missing=torch.tensor(mask,device=a.device)
                result['physical_link']=stats
            else:
                missing=erase_packets(ids,a.per)
                receiver_ids=ids.masked_fill(missing,0)
            labels=torch.tensor([ck['classes'] if a.label is None else a.label],device=a.device)
            restored=m.recover(receiver_ids,missing,labels)
            save_grid([im[0],vq.decode(ids)[0],vq.decode(restored)[0]],out/'reconstruction.png')
            result.update(per=a.per if a.link=='erasure' else None,link=a.link,label=a.label,missing_tokens=int(missing.sum()),
                          tokens=ids.shape[1],reconstruction='reconstruction.png')
        else: raise ValueError('Unknown checkpoint model')
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
