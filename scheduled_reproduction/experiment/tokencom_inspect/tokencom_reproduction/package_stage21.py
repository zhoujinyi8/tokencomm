"""Package stage 21 reproducibility evidence without the large checkpoint files."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
files = [
    (root / name, name) for name in (
        'fetch_official_weights.py', 'verify_stage21_official_vqgan.py',
        'verify_stage21_official_maskgit.py', 'smoke_stage21_official_pair.py',
        'runs/stage21_official_vqgan_comparison.json',
        'runs/stage21_official_maskgit_structure.json',
        'runs/stage21_official_pair_smoke.json')
] + [
    (outputs / name, name) for name in (
        'TokCom_第二十一阶段作者权重兼容性核验_2026-11-06.md',
        'TokCom_复现总览_截至2026-11-09.md')
]
manifest = {arc: hashlib.sha256(src.read_bytes()).hexdigest() for src, arc in files}
target = outputs / 'TokCom_stage21_official_weights_compatibility.zip'
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for src, arc in files:
        archive.write(src, arc)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + 1
print(target, target.stat().st_size)
