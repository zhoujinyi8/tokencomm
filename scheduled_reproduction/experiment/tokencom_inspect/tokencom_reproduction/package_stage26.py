"""Package author-token PHY development evidence without large weights/images."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
items = [(name, root) for name in (
    'cache_stage26_official_tokens.py', 'evaluate_stage26_official_token_link.py',
    'smoke_stage25_sionna_link.py',
    'runs/stage26_official_demo_tokens.npz',
    'runs/stage26_official_token_link_snr8_5trials.npz')]
items += [('TokCom_第二十六阶段作者Token物理链路开发集诊断_2026-11-12.md', outputs)]
target = outputs / 'TokCom_stage26_official_token_phy_dev.zip'
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(items) + 1
print(target, target.stat().st_size)
