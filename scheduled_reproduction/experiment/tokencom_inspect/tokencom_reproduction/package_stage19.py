"""Package coded feedback-header development diagnostics without model weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
files = {
    'digital_link.py': root / 'digital_link.py',
    'diagnose_stage16_feedback_retry.py': root / 'diagnose_stage16_feedback_retry.py',
    'smoke_stage13_physical_retransmission.py': root / 'smoke_stage13_physical_retransmission.py',
    'summarize_stage19_header.py': root / 'summarize_stage19_header.py',
    'summarize_stage19_snr.py': root / 'summarize_stage19_snr.py',
    'verify_stage19_header_live.py': root / 'verify_stage19_header_live.py',
    'package_stage19.py': Path(__file__),
    'stage19_sensitivity.json': root / 'runs/stage19_header16_sensitivity.json',
    'stage19_snr_sensitivity.json': root / 'runs/stage19_header_snr_sensitivity.json',
    'stage19_live10.json': root / 'runs/stage19_header16_live_verify10/results.json',
    'stage19_regression_default.json': root / 'runs/stage19_regression_default.json',
    'TokCom_第十九阶段反馈包头开销诊断_2026-11-04.md':
        out / 'TokCom_第十九阶段反馈包头开销诊断_2026-11-04.md',
    'TokCom_复现总览_截至2026-11-04.md':
        out / 'TokCom_复现总览_截至2026-11-04.md',
}
files.update({f'feedback_seed{s}.json': root / f'runs/stage19_header16_seed{s}.json'
              for s in range(20261029, 20261035)})
files.update({f'data_seed{s}.json': root / f'runs/stage14_random_trials_seed{s}/results.json'
              for s in range(20261029, 20261032)})
files.update({f'feedback_snr{snr}_header{header}.json':
              root / f'runs/stage19_snr{snr}_header{header}.json'
              for snr in (6, 10) for header in (0, 16)})
assert all(path.is_file() for path in files.values())
archive = out / 'TokCom_stage19_feedback_header_dev.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items():
        zipped.write(path, 'stage19/' + name)
    zipped.writestr('stage19/sha256_manifest.json', json.dumps({
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()}, indent=2))
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) + 1 == len(set(zipped.namelist()))
    manifest = json.loads(zipped.read('stage19/sha256_manifest.json'))
    assert all(hashlib.sha256(zipped.read('stage19/' + name)).hexdigest() == digest
               for name, digest in manifest.items())
print(archive, archive.stat().st_size, 'bytes,', len(files) + 1, 'unique entries')
