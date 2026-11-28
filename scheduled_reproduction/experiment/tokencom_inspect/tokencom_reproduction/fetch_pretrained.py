"""Fetch pinned public tokenizer and sample with SHA256 verification; requires curl."""
import argparse
import json
import subprocess
from pathlib import Path
from pretrained_vq import sha256,WEIGHT_SHA256

ROOT=Path(__file__).resolve().parent
RESOURCES=[{
    'path':'pretrained/maskgit-vqgan-imagenet-f16-256.bin',
    'sha256':WEIGHT_SHA256,
    'urls':[
        'https://huggingface.co/fun-research/TiTok/resolve/ab646ed225080a3acb7c78440a574d7f67f16fa7/maskgit-vqgan-imagenet-f16-256.bin',
        'https://hf-mirror.com/fun-research/TiTok/resolve/ab646ed225080a3acb7c78440a574d7f67f16fa7/maskgit-vqgan-imagenet-f16-256.bin']
}, {
    'path':'data/public_samples/dog.jpg',
    'sha256':'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a',
    'urls':['https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg']
}]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); manifest=[]
    for r in RESOURCES:
        destination=ROOT/r['path']
        if destination.is_file() and sha256(destination)==r['sha256']:
            print('Verified:',r['path']); manifest.append({**r,'verified':True}); continue
        if a.verify_only: raise RuntimeError('Missing or incorrect resource: '+r['path'])
        destination.parent.mkdir(parents=True,exist_ok=True)
        partial=destination.with_suffix(destination.suffix+'.part')
        for url in r['urls']:
            proc=subprocess.run(['curl','-f','-L','--connect-timeout','15','--max-time','600','-o',str(partial),url],check=False)
            if proc.returncode==0 and sha256(partial)==r['sha256']:
                partial.replace(destination); break
        else: raise RuntimeError('No verified download for '+r['path'])
        manifest.append({**r,'verified':True})
    (ROOT/'pretrained/resource_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__': main()
