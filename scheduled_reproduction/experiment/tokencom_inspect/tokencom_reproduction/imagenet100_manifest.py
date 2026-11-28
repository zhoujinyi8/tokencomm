"""Structural and leakage checks; a valid schema never proves author dataset provenance."""
import hashlib,json,re
from pathlib import PurePosixPath,Path
def check(condition,message):
    if not condition:raise ValueError(message)
def validate(manifest,official_synsets,forbidden_training_hashes=()):
    classes=manifest['classes'];images=manifest['images']
    check(len(classes)==100 and len(set(classes))==100,'exactly 100 unique classes required')
    check(all(re.fullmatch(r'n[0-9]{8}',v) and v in official_synsets for v in classes),'unknown synsets')
    check(bool(manifest.get('class_list_source')) and bool(manifest.get('split_source')),'source references required')
    paths=set();hash_splits={};counts={};test_classes=set();forbidden=set(forbidden_training_hashes)
    for row in images:
        name=row['path'];p=PurePosixPath(name);split=row['split'];h=row['sha256'];synset=row['synset']
        check(name and p.parts and '\\' not in name and ':' not in name and not p.is_absolute() and '..' not in p.parts,'unsafe relative path')
        check(name not in paths,'duplicate path');paths.add(name)
        check(split in ('train','validation','test') and synset in classes,'invalid split/class')
        check(bool(re.fullmatch(r'[0-9a-f]{64}',h)),'invalid SHA256')
        check(h not in hash_splits or hash_splits[h]==split,'same image content in different splits')
        hash_splits[h]=split
        if split=='train':check(h not in forbidden,'frozen evaluation image leaked into training')
        if split=='test':test_classes.add(synset)
        counts[split]=counts.get(split,0)+1
    check(test_classes==set(classes),'test split must cover all 100 declared classes')
    return dict(structural_checks_passed=True,class_count=100,image_count=len(images),split_counts=counts,
        ordered_class_list_sha256=hashlib.sha256(json.dumps(classes,separators=(',',':')).encode()).hexdigest(),
        paper_dataset_identity_established=False,external_author_provenance_review_required=True)
def verify_files(manifest,root):
    root=Path(root).resolve();verified=0
    for row in manifest['images']:
        p=(root/row['path']).resolve();check(p.is_relative_to(root),'path escapes dataset root')
        check(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],'image byte identity mismatch');verified+=1
    return verified
