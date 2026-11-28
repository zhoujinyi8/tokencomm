"""Verify a supplied manifest and local image bytes; author provenance remains a separate gate."""
import argparse,hashlib,json
from pathlib import Path
from imagenet100_manifest import validate,verify_files
p=argparse.ArgumentParser()
for key in ('manifest','root','synsets','output'):p.add_argument('--'+key,required=True)
p.add_argument('--forbidden-training-manifest',action='append',default=[]);a=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
synsets={line.split()[0] for line in Path(a.synsets).read_text(encoding='utf-8').splitlines() if line.strip()}
forbidden=set();forbidden_files={}
for path in a.forbidden_training_manifest:
    forbidden_files[path]=sha(path)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        for key in ('sha256','image_sha256'):
            if row.get(key):forbidden.add(row[key])
try:
    result=validate(manifest,synsets,forbidden);result['image_files_verified']=verify_files(manifest,a.root)
except (ValueError,KeyError,OSError) as exc:
    result=dict(structural_checks_passed=False,image_files_verified=0,error=str(exc),paper_dataset_identity_established=False)
result.update(manifest_sha256=sha(a.manifest),synset_mapping_sha256=sha(a.synsets),forbidden_manifest_sha256=forbidden_files,
              source_provenance_review_required=True,script_sha256=sha(__file__))
Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['structural_checks_passed'] else 2)
