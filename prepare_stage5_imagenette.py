"""Create a larger split that excludes every image used by the pilot experiment."""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from pretrained_vq import sha256


def pixel_digest(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb=image.convert('RGB')
        return hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest()


def reservoir(path,per_class,excluded,seed):
    source=pq.ParquetFile(path); meta=json.loads(source.schema_arrow.metadata[b'huggingface'])
    names=meta['info']['features']['label']['names']; rng=random.Random(seed)
    groups=[[] for _ in names]; seen=[0]*len(names); excluded_count=0
    for batch in source.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            raw=row['image']['bytes']; digest=hashlib.sha256(raw).hexdigest()
            if digest in excluded:excluded_count+=1;continue
            label=int(row['label']);seen[label]+=1;k=rng.randrange(seen[label])
            item=(raw,digest,row['image'].get('path'))
            if len(groups[label])<per_class:groups[label].append(item)
            elif k<per_class:groups[label][k]=item
    if any(len(g)!=per_class for g in groups):raise ValueError('Insufficient images')
    return groups,names,seen,excluded_count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-parquet',required=True);p.add_argument('--val-parquet',required=True)
    p.add_argument('--pilot',default='data/imagenette_pilot');p.add_argument('--out',default='data/imagenette_stage5')
    p.add_argument('--train-per-class',type=int,default=80);p.add_argument('--val-per-class',type=int,default=10)
    p.add_argument('--test-per-class',type=int,default=10);p.add_argument('--seed',type=int,default=20260928)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():p.error('Output manifest already exists')
    pilot=Path(a.pilot);old={s:[] for s in ['train','val','test']}
    for split in old:
        old[split]=[json.loads(x) for x in (pilot/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    old_train={r['sha256'] for r in old['train']};old_holdout={r['sha256'] for s in ['val','test'] for r in old[s]}
    train,names,train_seen,train_excluded=reservoir(a.train_parquet,a.train_per_class,old_train,a.seed)
    hold,names2,hold_seen,hold_excluded=reservoir(a.val_parquet,a.val_per_class+a.test_per_class,old_holdout,a.seed+1)
    if names!=names2:raise ValueError('Class vocab mismatch')
    groups={'train':train,'val':[g[:a.val_per_class] for g in hold],
            'test':[g[a.val_per_class:] for g in hold]}
    all_file=set();all_pixel=set();counts={}
    for split,class_groups in groups.items():
        records=[]
        for label,items in enumerate(class_groups):
            folder=out/'images'/split/str(label);folder.mkdir(parents=True,exist_ok=True)
            for raw,digest,source_path in items:
                pix=pixel_digest(raw)
                if digest in all_file or pix in all_pixel:raise ValueError('Cross-split duplicate')
                all_file.add(digest);all_pixel.add(pix)
                file=folder/(digest[:20]+'.jpg');file.write_bytes(raw)
                records.append({'image':file.relative_to(out).as_posix(),'label':label,
                    'question':'What is the main object in this image?','answer':names[label].lower(),
                    'sha256':digest,'pixel_sha256':pix,'source_path':source_path})
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        counts[split]=len(records)
    manifest={'dataset':'Imagenette 320px, stage-5 expanded disjoint subset','upstream':'https://github.com/fastai/imagenette',
        'mirror_repository':'https://huggingface.co/datasets/ShaomuTan/imagenette','seed':a.seed,'class_names':names,
        'split_counts':counts,'per_class':{'train':a.train_per_class,'val':a.val_per_class,'test':a.test_per_class},
        'excluded_pilot':{'train_hashes':len(old_train),'validation_and_test_hashes':len(old_holdout),
                          'source_train_rows_excluded':train_excluded,'source_validation_rows_excluded':hold_excluded},
        'source_available_after_exclusion':{'train':train_seen,'validation':hold_seen},
        'source_sha256':{'train':sha256(a.train_parquet),'validation':sha256(a.val_parquet)},
        'cross_split_file_duplicates':0,'cross_split_decoded_pixel_duplicates':0,
        'old_pilot_overlap':0,'test_policy':'Do not open until stage-5 configuration is frozen after validation.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
