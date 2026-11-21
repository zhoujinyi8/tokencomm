"""Real-image VQGAN + digital link, with optional trained matching MaskGIT recovery.

Without --maskgit this runs the explicit spatial nearest-neighbor baseline.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
from pretrained_vq import PretrainedVQGAN,DEFAULT_WEIGHT,WEIGHT_SHA256,sha256
from digital_link import transmit_tokens


def spatial_fill(received,missing):
    filled=received.copy(); size=math.isqrt(received.shape[1])
    if size*size!=received.shape[1]: raise ValueError('Non-square grid')
    xy=np.stack(np.unravel_index(np.arange(size*size),(size,size)),1)
    for b in range(received.shape[0]):
        known=np.flatnonzero(~missing[b]); lost=np.flatnonzero(missing[b])
        if not len(known):
            filled[b,lost]=0  # Explicit fallback when every packet is lost.
        elif len(lost):
            nearest=((xy[lost,None]-xy[known][None])**2).sum(-1).argmin(1)
            filled[b,lost]=received[b,known[nearest]]
    return filled


def save_image(tensor,path):
    arr=(tensor.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
    Image.fromarray(arr).save(path)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='data/public_samples/dog.jpg')
    p.add_argument('--weights',default=str(DEFAULT_WEIGHT))
    p.add_argument('--channel',choices=['awgn','rayleigh','noiseless'],default='awgn')
    p.add_argument('--snr',type=float,default=8.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--maskgit',help='Matched real-data checkpoint, e.g. runs/maskgit_imagenette/best.pt')
    p.add_argument('--label',type=int,help='Reliable receiver class index; omitted means no class side information')
    p.add_argument('--recovery-steps',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='runs/pretrained_vqgan')
    a=p.parse_args(); torch.set_num_threads(4)
    if a.recovery_steps<1: p.error('--recovery-steps must be positive')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with Image.open(a.image) as raw:
        image=ImageOps.fit(raw.convert('RGB'),(256,256),method=Image.Resampling.LANCZOS)
    im=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0).to(a.device)
    model=PretrainedVQGAN(a.weights).to(a.device)
    print('Loaded verified public VQGAN; encoding...',flush=True)
    ids=model.encode(im)
    if ids.shape!=(1,256): raise AssertionError('Tokenizer shape mismatch')
    clean=model.decode(ids)
    received,missing,stats=transmit_tokens(ids.cpu().numpy(),1024,a.snr,a.channel,a.seed)
    filled=spatial_fill(received,missing)
    recovery='spatial nearest-neighbor baseline'
    class_name=None
    if a.maskgit:
        from real_maskgit import load_real_maskgit
        checkpoint=torch.load(a.maskgit,map_location=a.device,weights_only=True)
        if checkpoint['identity']['tokenizer_sha256']!=WEIGHT_SHA256: raise ValueError('MaskGIT codebook/tokenizer mismatch')
        restorer=load_real_maskgit(checkpoint,a.device).eval()
        if a.label is not None and not 0<=a.label<restorer.classes: p.error('Label outside checkpoint class vocabulary')
        labels=torch.tensor([restorer.classes if a.label is None else a.label],device=a.device)
        filled=restorer.recover(torch.tensor(received,device=a.device),torch.tensor(missing,device=a.device),
                               labels,a.recovery_steps).cpu().numpy()
        recovery='trained compact MaskGIT on real Imagenette pilot'
        class_name=checkpoint['class_names'][a.label] if a.label is not None else None
    reconstructed=model.decode(torch.tensor(filled,device=a.device))
    def psnr(x): return -10*math.log10(max(float(F.mse_loss(x,im)),1e-12))
    save_image(im[0],out/'original.png'); save_image(clean[0],out/'clean_reconstruction.png')
    save_image(reconstructed[0],out/'channel_reconstruction.png')
    save_image(torch.cat([im[0],clean[0],reconstructed[0]],2),out/'comparison.png')
    np.savez_compressed(out/'tokens.npz',transmitted=ids.cpu().numpy(),received=received,missing=missing,filled=filled)
    result={'scope':'Pretrained VQGAN + digital link; recovery: '+recovery,
            'recovery_method':recovery,'receiver_class':class_name,
            'model_source':'fun-research/TiTok/maskgit-vqgan-imagenet-f16-256.bin',
            'weight_sha256':WEIGHT_SHA256,'input_sha256':sha256(a.image),'args':vars(a),
            'token_shape':list(ids.shape),'codebook_size':1024,'unique_tokens':int(ids.unique().numel()),
            'source_payload_bpp':256*10/(256*256),'coded_bpp':stats['coded_bits']/(256*256),
            'channel_uses_per_pixel':stats['symbols']/(256*256),
            'clean_psnr_db':psnr(clean),'channel_psnr_db':psnr(reconstructed),'link':stats,
            'overhead_exclusions':['CMI side channel','interleaver seed','framing','pilots','retransmissions']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
