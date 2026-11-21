"""Package ten-image five-trial author PSNR results."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage27_physical_reconstruction.py', 'batch_stage33_author_psnr.py',
         'summarize_stage33_psnr.py', 'runs/stage33_author_psnr_10images_5trials.json',
         'runs/stage26_official_demo_tokens.npz', 'runs/stage26_official_token_link_snr8_5trials.npz']
names += ['runs/stage31_author_psnr_summary.json']
names += [f'runs/stage31_author_metric_smoke_trial0_image{i}.json' for i in range(10)]
names += [f'runs/stage32_author_psnr_trial{i}_image0.json' for i in range(1, 5)]
names += [f'runs/stage33_author_psnr_trial{t}_image{i}.json' for t in range(1, 5) for i in range(1, 10)]
items = [(name, root) for name in names]
items += [('TokCom_第三十三阶段十图五轮原图PSNR_2026-11-18.md', outputs)]
target = outputs / 'TokCom_stage33_original_psnr_10images_5trials.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size, len(items))
