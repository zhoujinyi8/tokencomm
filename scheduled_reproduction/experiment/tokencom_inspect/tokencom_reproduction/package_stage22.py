"""Package the author's multistep single-image smoke evidence."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage21_official_pair.py',
         'runs/stage22_official_multistep_smoke.json',
         'runs/stage22_official_multistep_seed20261110.json',
         'runs/stage22_official_multistep_seed20261111.json']
files = [(root / name, name) for name in names]
report = 'TokCom_第二十二阶段作者多步解码单图复核_2026-11-10.md'
files.append((outputs / report, report))
target = outputs / 'TokCom_stage22_official_multistep_smoke.zip'
manifest = {arc: hashlib.sha256(path.read_bytes()).hexdigest() for path, arc in files}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path, arc in files: archive.write(path, arc)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + 1
print(target, target.stat().st_size)
