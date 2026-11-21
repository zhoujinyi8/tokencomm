"""Package reproducible development feedback retry diagnostics without model weights."""
import json
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent
out_dir = root.parents[2] / 'outputs'
archive = out_dir / 'TokCom_stage16_feedback_retry_dev.zip'
files = {
    name: root / name for name in [
        'digital_link.py', 'smoke_stage13_physical_retransmission.py',
        'diagnose_stage16_feedback_retry.py', 'summarize_stage16_feedback_retry.py',
        'verify_stage16_live_replay.py', 'package_stage16.py', 'README.md']}
files.update({
    'stage16_summary.json': root / 'runs/stage16_feedback_retry_summary.json',
    'stage16_sensitivity.json': root / 'runs/stage16_feedback_retry_sensitivity.json',
    'stage16_one_attempt_validation.json': root / 'runs/stage16_one_attempt_validation.json',
    'stage16_three_attempt_tradeoff.json': root / 'runs/stage16_three_attempt_tradeoff.json',
    'stage16_live_verify10.json': root / 'runs/stage16_live_feedback_verify10/results.json',
    'stage16_live_verify100.json': root / 'runs/stage16_live_feedback_verify100/results.json',
    'TokCom_第十六阶段反馈重发与实链路核验_2026-10-29.md':
        out_dir / 'TokCom_第十六阶段反馈重发与实链路核验_2026-10-29.md',
    'TokCom_复现总览_截至2026-10-29.md':
        out_dir / 'TokCom_复现总览_截至2026-10-29.md',
})
files.update({f'data_seed{s}.json': root / f'runs/stage14_random_trials_seed{s}/results.json'
              for s in range(20261029, 20261032)})
files.update({f'feedback_seed{s}.json': root / f'runs/stage16_feedback_retry_seed{s}.json'
              for s in range(20261030, 20261035)})
assert all(path.is_file() for path in files.values())
assert len(files) == len(set(files))
for seed in range(20261029, 20261032):
    case = json.loads(files[f'data_seed{seed}.json'].read_text(encoding='utf-8'))
    assert case['seed'] == seed and len(case['rows']) == 100
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items():
        zipped.write(path, 'stage16/' + name)
    zipped.writestr('stage16/sha256_manifest.json', json.dumps({
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()}, indent=2))
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) + 1 == len(set(zipped.namelist()))
    manifest = json.loads(zipped.read('stage16/sha256_manifest.json'))
    assert all(hashlib.sha256(zipped.read('stage16/' + name)).hexdigest() == digest
               for name, digest in manifest.items())
print(archive, archive.stat().st_size, 'bytes,', len(files) + 1, 'unique entries')
