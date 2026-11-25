"""Check saved project-disjoint holdout identities without model inference."""
import json
from collections import Counter
from pathlib import Path
from prepare_stage10_holdout import prior_hashes
from prepare_stage5_imagenette import pixel_digest
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
folder=root/'data/imagenette_stage44_holdout'
protocol_path=root/'runs/stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(folder/'test.jsonl')==manifest['test_jsonl_sha256']
prior=[root/p for p in protocol['excluded_datasets']]
files,pixels=prior_hashes(prior)
for path in (root/'data/public_samples').rglob('*'):
    if path.suffix.lower() in ('.jpg','.jpeg','.png'):
        files.add(sha256(path));pixels.add(pixel_digest(path.read_bytes()))
records=[json.loads(s) for s in (folder/'test.jsonl').read_text(encoding='utf-8').splitlines()]
counts=Counter();seen_files=set();seen_pixels=set()
for record in records:
    path=folder/record['image']
    digest=sha256(path);pixel_hash=pixel_digest(path.read_bytes())
    assert digest==record['sha256'] and pixel_hash==record['pixel_sha256']
    assert digest not in files|seen_files and pixel_hash not in pixels|seen_pixels
    seen_files.add(digest);seen_pixels.add(pixel_hash);counts[record['label']]+=1
assert len(records)==100 and sorted(counts.values())==[10]*10
assert not (folder/'token_cache').exists()
for name,digest in protocol['weight_sha256'].items():assert sha256(root/name)==digest
for name,digest in protocol['script_sha256'].items():assert sha256(root/name)==digest
result={'count':100,'class_counts':dict(counts),'prior_unique_files':len(files),'prior_unique_pixels':len(pixels),
        'file_overlap':0,'pixel_overlap':0,'within_holdout_duplicates':0,
        'model_inference_performed':False,'protocol_sha256':sha256(protocol_path),
        'test_jsonl_sha256':sha256(folder/'test.jsonl')}
(root/'runs/stage44_holdout_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
