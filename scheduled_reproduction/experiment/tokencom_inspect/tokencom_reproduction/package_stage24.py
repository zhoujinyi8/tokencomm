"""Package paired preprocessing evidence without checkpoint files."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
items = [('evaluate_stage23_official_dev.py', root),
         ('runs/stage23_official_multiclass_dev.json', root),
         ('runs/stage24_official_demo_resize_dev.json', root),
         ('TokCom_第二十三阶段作者权重十类开发集诊断_2026-11-10.md', outputs),
         ('TokCom_第二十四阶段作者演示预处理配对核验_2026-11-10.md', outputs)]
target = outputs / 'TokCom_stage24_official_preprocessing_pair.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
print(target, target.stat().st_size)
