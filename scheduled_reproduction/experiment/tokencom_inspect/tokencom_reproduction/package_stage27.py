"""Package one-image paired physical reconstruction evidence."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
items = [(name, root) for name in (
    'smoke_stage27_physical_reconstruction.py',
    'runs/stage27_physical_reconstruction_one_image.json',
    'runs/stage26_official_token_link_snr8_5trials.npz')]
items += [('TokCom_第二十七阶段作者链路单图重建核验_2026-11-12.md', outputs)]
target = outputs / 'TokCom_stage27_physical_reconstruction_smoke.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
print(target, target.stat().st_size)
