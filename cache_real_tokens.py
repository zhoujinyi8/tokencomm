"""Cache real pretrained VQGAN tokens, with dataset identity and resumable per-image files."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from pretrained_vq import PretrainedVQGAN,WEIGHT_SHA256,sha256


def read_records(root,split):
    return [json.loads(s) for s in (Path(root)/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]


def load_image(path):
    with Image.open(path) as im:
        im=ImageOps.fit(im.convert('RGB'),(256,256),method=Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).permute(2,0,1)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_pilot'); p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test'])
    a=p.parse_args(); torch.set_num_threads(a.threads)
    root=Path(a.data); out=root/'token_cache'; out.mkdir(exist_ok=True)
    meta=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    identity={'tokenizer_sha256':WEIGHT_SHA256,'preprocessing':'PIL ImageOps.fit 256 LANCZOS RGB float[0,1]',
              'manifest_sha256':sha256(root/'manifest.json')}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
        raise ValueError('Cache identity mismatch; use a new dataset directory')
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    model=PretrainedVQGAN().to(a.device); start=time.monotonic(); count=0
    for split in a.splits:
        records=read_records(root,split); tokens=[]
        for i,row in enumerate(records):
            path=root/row['image']
            if sha256(path)!=row['sha256']: raise ValueError('Image contents changed: '+str(path))
            cached=out/(row['sha256']+'.npy')
            try:
                ids=np.load(cached,allow_pickle=False) if cached.exists() else None
                if ids is not None and (ids.shape!=(256,) or ids.min()<0 or ids.max()>=1024): ids=None
            except (OSError,ValueError): ids=None
            if ids is None:
                ids=model.encode(load_image(path)[None].to(a.device))[0].cpu().numpy().astype(np.int16)
                temporary=cached.with_suffix(cached.suffix+'.tmp')
                with temporary.open('wb') as stream: np.save(stream,ids,allow_pickle=False)
                temporary.replace(cached)
            if ids.shape!=(256,) or ids.min()<0 or ids.max()>=1024: raise ValueError('Invalid token cache')
            tokens.append(torch.tensor(ids,dtype=torch.long)); count+=1
            if i%10==0 or i==len(records)-1:
                print(f'{split} {i+1}/{len(records)} elapsed={time.monotonic()-start:.1f}s',flush=True)
        torch.save({'ids':torch.stack(tokens),'labels':torch.tensor([r['label'] for r in records]),
                    'records':records,'class_names':meta['class_names'],'identity':identity},out/f'{split}.pt')
    (out/'summary.json').write_text(json.dumps({'images':count,'splits':a.splits,'test_opened':'test' in a.splits,'elapsed_seconds':time.monotonic()-start,
        'torch':str(torch.__version__),'device':a.device,'identity':identity},indent=2),encoding='utf-8')


if __name__=='__main__': main()
