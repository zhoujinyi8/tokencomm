"""Build a larger Imagenette train split while reusing the locked Stage-7 validation set.

The Stage-7 test manifest and images are deliberately never read by this script.
"""
import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from pretrained_vq import sha256


def pixel_digest(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb=image.convert('RGB')
        return hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest()


def balanced_reservoir(parquet_path, per_class, seed):
    source=pq.ParquetFile(parquet_path)
    meta=json.loads(source.schema_arrow.metadata[b'huggingface'])
    names=meta['info']['features']['label']['names']; rng=random.Random(seed)
    groups=[[] for _ in names]; seen=[0]*len(names)
    for arrow_batch in source.iter_batches(batch_size=128):
        for row in arrow_batch.to_pylist():
            label=int(row['label']); seen[label]+=1; k=rng.randrange(seen[label])
            raw=row['image']['bytes']; item=(raw,hashlib.sha256(raw).hexdigest(),row['image'].get('path'))
            if len(groups[label])<per_class: groups[label].append(item)
            elif k<per_class: groups[label][k]=item
    if any(len(group)!=per_class for group in groups):
        raise ValueError(f'Insufficient source rows: {seen}')
    return groups,names,seen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-parquet',required=True)
    p.add_argument('--stage7',default='data/imagenette_stage7')
    p.add_argument('--out',default='data/imagenette_stage10')
    p.add_argument('--train-per-class',type=int,default=800)
    p.add_argument('--seed',type=int,default=20261003)
    a=p.parse_args(); source=Path(a.stage7); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): p.error('Output manifest already exists')

    groups,names,seen=balanced_reservoir(a.train_parquet,a.train_per_class,a.seed)
    train=[]; train_hashes=set(); train_pixels=set()
    for label,items in enumerate(groups):
        folder=out/'images'/'train'/str(label); folder.mkdir(parents=True,exist_ok=True)
        for raw,digest,source_path in items:
            pix=pixel_digest(raw)
            if digest in train_hashes or pix in train_pixels: raise ValueError('Duplicate training image')
            train_hashes.add(digest); train_pixels.add(pix)
            file=folder/(digest[:20]+'.jpg'); file.write_bytes(raw)
            train.append({'image':file.relative_to(out).as_posix(),'label':label,
                          'question':'What is the main object in this image?',
                          'answer':names[label].lower(),'sha256':digest,
                          'pixel_sha256':pix,'source_path':source_path})

    # Reuse only Stage-7 validation. Do not enumerate or read its test manifest.
    val=[]
    for line in (source/'val.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['sha256'] in train_hashes or row['pixel_sha256'] in train_pixels:
            raise ValueError('Train/validation overlap')
        src=source/row['image']; dst=out/'images'/'val'/str(row['label'])/src.name
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        copied=dict(row); copied['image']=dst.relative_to(out).as_posix(); val.append(copied)

    for split,records in [('train',train),('val',val)]:
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as stream:
            for row in records: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    stage7_manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest={'dataset':'Imagenette 320px, Stage-10 expanded training split',
              'upstream':'https://github.com/fastai/imagenette','seed':a.seed,
              'class_names':names,'split_counts':{'train':len(train),'val':len(val)},
              'per_class':{'train':a.train_per_class,'val':len(val)//len(names)},
              'source_available':seen,'source_train_sha256':sha256(a.train_parquet),
              'validation_origin':'Exact copy of Stage-7 validation records and images',
              'stage7_manifest_sha256':sha256(source/'manifest.json'),
              'stage7_validation_count':stage7_manifest['split_counts']['val'],
              'cross_split_file_duplicates':0,'cross_split_decoded_pixel_duplicates':0,
              'test_opened':False,
              'test_policy':'Stage-7 test manifest and images were not read or copied.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
