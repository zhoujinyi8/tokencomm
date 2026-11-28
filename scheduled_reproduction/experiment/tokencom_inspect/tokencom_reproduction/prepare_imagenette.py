"""Build deterministic, class-balanced disjoint image splits from public Parquet files."""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def sample_file(path,per_class,seed):
    source=pq.ParquetFile(path)
    metadata=json.loads(source.schema_arrow.metadata[b'huggingface'])
    names=metadata['info']['features']['label']['names']
    rng=random.Random(seed); seen=[0]*len(names); groups=[[] for _ in names]
    for batch in source.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            label=row['label']; seen[label]+=1
            k=rng.randrange(seen[label])
            if len(groups[label])<per_class: groups[label].append(row)
            elif k<per_class: groups[label][k]=row
    if any(len(g)!=per_class for g in groups): raise ValueError('Not enough images per class')
    return groups,names,seen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-parquet',required=True); p.add_argument('--val-parquet',required=True)
    p.add_argument('--train-per-class',type=int,default=24)
    p.add_argument('--val-per-class',type=int,default=4)
    p.add_argument('--test-per-class',type=int,default=4)
    p.add_argument('--out',default='data/imagenette_pilot')
    p.add_argument('--seed',type=int,default=20260927)
    a=p.parse_args()
    if min(a.train_per_class,a.val_per_class,a.test_per_class)<1: p.error('Split sizes must be positive')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): p.error('Output manifest exists; use a new --out to preserve splits')
    train,names,train_count=sample_file(a.train_parquet,a.train_per_class,a.seed)
    holdout,other,holdout_count=sample_file(a.val_parquet,a.val_per_class+a.test_per_class,a.seed+1)
    if names!=other: raise ValueError('Class vocab mismatch')
    groups={'train':train,'val':[g[:a.val_per_class] for g in holdout],
            'test':[g[a.val_per_class:] for g in holdout]}
    manifest={'dataset':'Imagenette 320px, 10 classes, pilot subset',
              'upstream':'https://github.com/fastai/imagenette',
              'mirror_repository':'https://huggingface.co/datasets/ShaomuTan/imagenette',
              'seed':a.seed,'class_names':names,'source_train_counts':train_count,
              'source_validation_counts':holdout_count,'source_sha256':{
                  'train_parquet':digest(a.train_parquet),'validation_parquet':digest(a.val_parquet)},
              'args':vars(a),'split_counts':{},'selection':'Per-class reservoir sampling; validation/test disjoint subsets of official validation',
              'pretraining_caveat':'Tokenizer is pretrained on ImageNet; this is not a from-scratch or unseen-pretraining-data benchmark.'}
    hashes=set(); pixel_hashes=set()
    for split,by_class in groups.items():
        records=[]
        for label,rows in enumerate(by_class):
            folder=out/'images'/split/str(label); folder.mkdir(parents=True,exist_ok=True)
            for row in rows:
                raw=row['image']['bytes']; h=hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as im:
                    im=im.convert('RGB'); ph=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                if h in hashes or ph in pixel_hashes: raise ValueError('Duplicate image detected; select another seed')
                hashes.add(h); pixel_hashes.add(ph)
                file=folder/(h[:20]+'.jpg'); file.write_bytes(raw)
                records.append({'image':file.relative_to(out).as_posix(),'label':label,
                                'question':'What is the main object in this image?',
                                'answer':names[label].lower(),'sha256':h,'pixel_sha256':ph,
                                'source_path':row['image'].get('path')})
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for row in records: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        manifest['split_counts'][split]=len(records)
    manifest['cross_split_duplicates']=0
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
