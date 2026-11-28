"""Package the paired LPIPS range audit, excluding large weights and images."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
names = ['batch_stage34_lpips.py', 'prepare_stage34_lpips.py',
         'summarize_stage34_five_trials.py', 'summarize_stage34_lpips.py',
         'runs/stage34_lpips_five_trials_summary.json',
         'runs/stage34_lpips_summary.json',
         'runs/stage26_official_demo_tokens.npz',
         'runs/stage26_official_token_link_snr8_5trials.npz']
names += [f'runs/stage34_lpips_trial{t}_image{i}.json' for t in range(5) for i in range(10)]
items = [(name, root) for name in names]
items += [('TokCom_第三十四阶段十图五轮LPIPS输入范围诊断_2026-11-19.md', outputs)]
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
target = outputs / 'TokCom_stage34_lpips_10images_5trials.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(target, target.stat().st_size)
