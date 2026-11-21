"""Package the pinned official-source audit and progress notes."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
files = {
    'audit_stage20_official.py': root / 'audit_stage20_official.py',
    'package_stage20.py': Path(__file__),
    'stage20_official_source_audit.json': root / 'runs/stage20_official_source_audit.json',
    'TokCom_第二十阶段官方实现对照审计_2026-11-04.md':
        out / 'TokCom_第二十阶段官方实现对照审计_2026-11-04.md',
    'TokCom_复现总览_截至2026-11-04.md':
        out / 'TokCom_复现总览_截至2026-11-04.md',
}
assert all(p.is_file() for p in files.values())
archive = out / 'TokCom_stage20_official_source_audit.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items():
        zipped.write(path, 'stage20/' + name)
    zipped.writestr('stage20/sha256_manifest.json', json.dumps({
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()}, indent=2))
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) + 1 == len(set(zipped.namelist()))
print(archive, archive.stat().st_size, 'bytes,', len(files) + 1, 'unique entries')
