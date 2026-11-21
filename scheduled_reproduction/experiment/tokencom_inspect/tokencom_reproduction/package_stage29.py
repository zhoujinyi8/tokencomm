"""Package the single-image five-channel-trial check."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage27_physical_reconstruction.py',
         'runs/stage27_physical_reconstruction_one_image.json']
names += [f'runs/stage29_physical_reconstruction_trial{i}_image0.json' for i in range(1, 5)]
items = [(name, root) for name in names]
items += [('TokCom_第二十九阶段单图五轮信道波动抽查_2026-11-12.md', outputs),
          ('TokCom_复现总览_截至2026-11-12_阶段29.md', outputs)]
target = outputs / 'TokCom_stage29_one_image_5trials.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size)
