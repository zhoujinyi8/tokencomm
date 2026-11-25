"""Package stage17 development feature diagnostics without third-party weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
files = {
    'diagnose_stage17_resnet_feature.py': root / 'diagnose_stage17_resnet_feature.py',
    'analyze_stage17_resnet_feature.py': root / 'analyze_stage17_resnet_feature.py',
    'report_stage17.py': root / 'report_stage17.py',
    'update_stage17_overview.py': root / 'update_stage17_overview.py',
    'package_stage17.py': Path(__file__),
    'stage17_results.json': root / 'runs/stage17_resnet_feature_full/results.json',
    'stage17_analysis.json': root / 'runs/stage17_resnet_feature_full/analysis.json',
    'stage11_reference_results.json': root / 'runs/stage11_pretrained_reconstruction_full/results.json',
    'TokCom_第十七阶段图像特征相似度开发验证_2026-11-01.md':
        out / 'TokCom_第十七阶段图像特征相似度开发验证_2026-11-01.md',
    'TokCom_复现总览_截至2026-11-01.md':
        out / 'TokCom_复现总览_截至2026-11-01.md',
}
assert all(path.is_file() for path in files.values())
archive = out / 'TokCom_stage17_resnet_feature_dev.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items():
        zipped.write(path, 'stage17/' + name)
    zipped.writestr('stage17/sha256_manifest.json', json.dumps({
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()}, indent=2))
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) + 1 == len(set(zipped.namelist()))
    manifest = json.loads(zipped.read('stage17/sha256_manifest.json'))
    assert all(hashlib.sha256(zipped.read('stage17/' + name)).hexdigest() == digest
               for name, digest in manifest.items())
print(archive, archive.stat().st_size, 'bytes,', len(files) + 1, 'unique entries')
