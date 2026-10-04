"""Build a third disjoint Imagenette split for ContextMaskGITv2 development."""
import argparse,hashlib,io,json,random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from pretrained_vq import sha256

def pixel(raw):
    with Image.open(io.BytesIO(raw)) as im:
        rgb=im.convert('RGB');return hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest()

def old_hashes(roots,split):
    values=set()
    for root in roots:
        p=root/f'{split}.jsonl'
        if p.exists():values.update(json.loads(x)['sha256'] for x in p.read_text(encoding='utf-8').splitlines() if x)
    return values

def sample(path,n,exclude,seed):
    pqf=pq.ParquetFile(path);meta=json.loads(pqf.schema_arrow.metadata[b'huggingface']);names=meta['info']['features']['label']['names']
    groups=[[] for _ in names];seen=[0]*len(names);rng=random.Random(seed)
    for batch in pqf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            raw=row['image']['bytes'];h=hashlib.sha256(raw).hexdigest()
            if h in exclude:continue
            label=int(row['label']);seen[label]+=1;k=rng.randrange(seen[label]);item=(raw,h,row['image'].get('path'))
            if len(groups[label])<n:groups[label].append(item)
            elif k<n:groups[label][k]=item
    if any(len(x)!=n for x in groups):raise ValueError('Insufficient samples after exclusions')
    return groups,names,seen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train-parquet',required=True);p.add_argument('--val-parquet',required=True)
    p.add_argument('--out',default='data/imagenette_stage7');p.add_argument('--seed',type=int,default=20260929)
    p.add_argument('--train-per-class',type=int,default=100);p.add_argument('--val-per-class',type=int,default=15);p.add_argument('--test-per-class',type=int,default=15)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():p.error('Output already exists')
    roots=[Path('data/imagenette_pilot'),Path('data/imagenette_stage5')]
    tr_ex=old_hashes(roots,'train');ho_ex=old_hashes(roots,'val')|old_hashes(roots,'test')
    train,names,ts=sample(a.train_parquet,a.train_per_class,tr_ex,a.seed)
    hold,names2,vs=sample(a.val_parquet,a.val_per_class+a.test_per_class,ho_ex,a.seed+1)
    if names!=names2:raise ValueError('Class vocabulary mismatch')
    groups={'train':train,'val':[x[:a.val_per_class] for x in hold],'test':[x[a.val_per_class:] for x in hold]}
    file_seen=set();pixel_seen=set();counts={}
    for split,classes in groups.items():
        rows=[]
        for label,items in enumerate(classes):
            folder=out/'images'/split/str(label);folder.mkdir(parents=True,exist_ok=True)
            for raw,h,path in items:
                px=pixel(raw)
                if h in file_seen or px in pixel_seen:raise ValueError('Duplicate across stage-7 splits')
                file_seen.add(h);pixel_seen.add(px);dst=folder/(h[:20]+'.jpg');dst.write_bytes(raw)
                rows.append({'image':dst.relative_to(out).as_posix(),'label':label,'question':'What is the main object in this image?',
                             'answer':names[label].lower(),'sha256':h,'pixel_sha256':px,'source_path':path})
        (out/f'{split}.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8');counts[split]=len(rows)
    manifest={'dataset':'Imagenette stage-7 disjoint subset','seed':a.seed,'class_names':names,'split_counts':counts,
      'per_class':{'train':a.train_per_class,'val':a.val_per_class,'test':a.test_per_class},
      'excluded_previous':{'roots':[str(x) for x in roots],'train_hashes':len(tr_ex),'holdout_hashes':len(ho_ex)},
      'cross_split_file_duplicates':0,'cross_split_decoded_pixel_duplicates':0,'test_policy':'Unused until v2 configuration is frozen.',
      'source_sha256':{'train':sha256(a.train_parquet),'validation':sha256(a.val_parquet)}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
