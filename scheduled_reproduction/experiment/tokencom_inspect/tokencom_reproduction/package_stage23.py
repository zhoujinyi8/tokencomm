"""Package stage 23 code and results, excluding pretrained weights and images."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
files = [
    (root / 'evaluate_stage23_official_dev.py', 'evaluate_stage23_official_dev.py'),
    (root / 'runs/stage23_official_multiclass_dev.json', 'runs/stage23_official_multiclass_dev.json'),
    (outputs / 'TokCom_第二十三阶段作者权重十类开发集诊断_2026-11-10.md',
     'TokCom_第二十三阶段作者权重十类开发集诊断_2026-11-10.md'),
]
target = outputs / 'TokCom_stage23_official_multiclass_dev.zip'
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path, name in files: archive.write(path, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + 1
print(target, target.stat().st_size)
