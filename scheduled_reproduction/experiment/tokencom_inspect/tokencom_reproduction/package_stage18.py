"""Package paired metric diagnostics and selected visual examples."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
files = {
    'analyze_stage18_cross_metric.py': root / 'analyze_stage18_cross_metric.py',
    'render_stage18_discordant.py': root / 'render_stage18_discordant.py',
    'package_stage18.py': Path(__file__),
    'stage18_results.json': root / 'runs/stage18_cross_metric/results.json',
    'stage11_reference_results.json': root / 'runs/stage11_pretrained_reconstruction_full/results.json',
    'stage17_reference_results.json': root / 'runs/stage17_resnet_feature_full/results.json',
    'TokCom_stage18_psnr_feature_scatter.png': out / 'TokCom_stage18_psnr_feature_scatter.png',
    'TokCom_stage18_metric_discordance_montage.png': out / 'TokCom_stage18_metric_discordance_montage.png',
    'TokCom_第十八阶段跨指标一致性诊断_2026-11-02.md':
        out / 'TokCom_第十八阶段跨指标一致性诊断_2026-11-02.md',
    'TokCom_复现总览_截至2026-11-02.md':
        out / 'TokCom_复现总览_截至2026-11-02.md',
}
assert all(path.is_file() for path in files.values())
archive = out / 'TokCom_stage18_cross_metric_dev.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items():
        zipped.write(path, 'stage18/' + name)
    zipped.writestr('stage18/sha256_manifest.json', json.dumps({
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()}, indent=2))
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) + 1 == len(set(zipped.namelist()))
    manifest = json.loads(zipped.read('stage18/sha256_manifest.json'))
    assert all(hashlib.sha256(zipped.read('stage18/' + name)).hexdigest() == digest
               for name, digest in manifest.items())
print(archive, archive.stat().st_size, 'bytes,', len(files) + 1, 'unique entries')
