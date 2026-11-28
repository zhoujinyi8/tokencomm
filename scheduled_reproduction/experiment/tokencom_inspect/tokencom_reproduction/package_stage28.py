"""Package paired ten-image author reconstruction evidence without weights."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage27_physical_reconstruction.py',
         'summarize_stage28_physical_pair.py',
         'runs/stage26_official_token_link_snr8_5trials.npz',
         'runs/stage27_physical_reconstruction_one_image.json',
         'runs/stage27_physical_reconstruction_image1.json',
         'runs/stage28_physical_reconstruction_pair_summary.json']
names += [f'runs/stage28_physical_reconstruction_image{i}.json' for i in range(2, 10)]
items = [(name, root) for name in names]
items += [('TokCom_第二十八阶段作者物理接收十图配对重建_2026-11-12.md', outputs),
          ('TokCom_复现总览_截至2026-11-12_阶段28.md', outputs)]
target = outputs / 'TokCom_stage28_physical_pair_10dev.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
print(target, target.stat().st_size)
