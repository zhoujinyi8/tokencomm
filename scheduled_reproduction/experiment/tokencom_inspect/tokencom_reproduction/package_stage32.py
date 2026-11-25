"""Package one-image five-trial author-style PSNR evidence."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage27_physical_reconstruction.py', 'summarize_stage32_one_image_psnr.py',
         'runs/stage31_author_metric_smoke_trial0_image0.json',
         'runs/stage32_one_image_psnr_summary.json']
names += [f'runs/stage32_author_psnr_trial{i}_image0.json' for i in range(1, 5)]
items = [(name, root) for name in names]
items += [('TokCom_第三十二阶段单图五轮原图PSNR核验_2026-11-13.md', outputs)]
target = outputs / 'TokCom_stage32_one_image_5trial_psnr.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size)
