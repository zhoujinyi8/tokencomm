"""Download the exact public Parquet files used in the pilot, with SHA256 verification."""
import argparse
import hashlib
import subprocess
from pathlib import Path

FILES={
 'train':('train-00000-of-00001.parquet','b13d180724c25fd9daf01f6c10f35148d0aa855d9fe6ad96130fe71e65b77c9f'),
 'validation':('validation-00000-of-00001.parquet','fa51906de9c2847cd058d9896d6b76e97a4499e4e07efca14fdffb3ae2661847')}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='work/downloads')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for split,(file,digest) in FILES.items():
        dest=out/f'imagenette_{split}.parquet'
        if dest.exists() and sha(dest)==digest:
            print('Verified',dest); continue
        temporary=dest.with_suffix('.part')
        for host in ['huggingface.co','hf-mirror.com']:
            url=f'https://{host}/datasets/ShaomuTan/imagenette/resolve/main/320px/{file}'
            result=subprocess.run(['curl','-f','-L','--connect-timeout','15','--max-time','600','-o',str(temporary),url],check=False)
            if result.returncode==0 and sha(temporary)==digest:
                temporary.replace(dest); print('Downloaded and verified',dest); break
        else: raise RuntimeError('No matching download for '+split)


if __name__=='__main__': main()
