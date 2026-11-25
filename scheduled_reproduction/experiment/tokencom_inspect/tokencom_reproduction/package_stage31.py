"""Package author-style original-image PSNR diagnostics without model weights/images."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['smoke_stage27_physical_reconstruction.py', 'run_stage31_original_psnr.py',
         'summarize_stage31_author_psnr.py', 'runs/stage31_author_psnr_summary.json',
         'runs/stage26_official_demo_tokens.npz',
         'runs/stage26_official_token_link_snr8_5trials.npz']
names += [f'runs/stage31_author_metric_smoke_trial0_image{i}.json' for i in range(10)]
items = [(name, root) for name in names]
items += [('TokCom_第三十一阶段作者图像指标口径与原图PSNR_2026-11-13.md', outputs)]
target = outputs / 'TokCom_stage31_original_image_psnr_10dev.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size, len(items))
