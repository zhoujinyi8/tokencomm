"""Package ten-image five-trial author reconstruction evidence without weights."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
run = root / 'runs'
names = ['smoke_stage27_physical_reconstruction.py',
         'summarize_stage30_multitrial_progress.py',
         'runs/stage26_official_token_link_snr8_5trials.npz',
         'runs/stage30_multitrial_progress.json',
         'runs/stage27_physical_reconstruction_one_image.json',
         'runs/stage27_physical_reconstruction_image1.json']
names += [f'runs/stage28_physical_reconstruction_image{i}.json' for i in range(2, 10)]
names += [f'runs/stage29_physical_reconstruction_trial{t}_image0.json' for t in range(1, 5)]
names += [f'runs/stage30_physical_reconstruction_trial{t}_image{i}.json'
          for t in range(1, 5) for i in range(1, 10)]
items = [(name, root) for name in names]
items += [('TokCom_第三十阶段作者十图五轮物理接收重建_2026-11-12.md', outputs)]
target = outputs / 'TokCom_stage30_10images_5trials.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size, len(items))
