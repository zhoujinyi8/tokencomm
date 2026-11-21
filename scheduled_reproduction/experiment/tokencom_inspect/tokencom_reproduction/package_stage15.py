"""Package feedback-gated development diagnostics and inputs."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
archive = out / 'TokCom_stage15_feedback_gate_dev.zip'
files = {
    'smoke_stage13_physical_retransmission.py': root / 'smoke_stage13_physical_retransmission.py',
    'summarize_stage14_random_trials.py': root / 'summarize_stage14_random_trials.py',
    'diagnose_stage15_feedback_gate.py': root / 'diagnose_stage15_feedback_gate.py',
    'summarize_stage15_feedback_sensitivity.py': root / 'summarize_stage15_feedback_sensitivity.py',
    'package_stage15.py': Path(__file__),
    'stage14_summary.json': root / 'runs/stage14_random_trials_summary.json',
    'stage15_summary.json': root / 'runs/stage15_feedback_gate_summary.json',
    'feedback_sensitivity.json': root / 'runs/stage15_feedback_sensitivity.json',
    'TokCom_第十五阶段反馈控制包诊断_2026-10-29.md':
        out / 'TokCom_第十五阶段反馈控制包诊断_2026-10-29.md',
    'TokCom_复现总览_截至2026-10-29.md':
        out / 'TokCom_复现总览_截至2026-10-29.md',
    **{f'seed{seed}.json': root / f'runs/stage14_random_trials_seed{seed}/results.json'
       for seed in range(20261029, 20261032)},
    **{f'feedback_seed{seed}.json': root / f'runs/stage15_feedback_gate_seed{seed}.json'
       for seed in range(20261030, 20261035)},
}
for seed in range(20261029, 20261032):
    case = json.loads(files[f'seed{seed}.json'].read_text(encoding='utf-8'))
    assert case['seed'] == seed and len(case['rows']) == 100
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, path in files.items(): zipped.write(path, 'stage15/' + name)
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) == len(set(zipped.namelist()))
print(archive, archive.stat().st_size, 'bytes,', len(files), 'unique entries')
