"""Package Sionna link source and evidence without the isolated environment."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
items = [('smoke_stage25_sionna_link.py', root),
         ('evaluate_stage25_sionna_multiseed.py', root),
         ('runs/stage25_sionna_link_smoke_v2.json', root),
         ('runs/stage25_sionna_multiseed20.json', root),
         ('TokCom_第二十五阶段作者物理链路包边界核验_2026-11-11.md', outputs)]
target = outputs / 'TokCom_stage25_sionna_link_smoke.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
print(target, target.stat().st_size)
