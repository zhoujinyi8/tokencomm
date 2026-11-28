"""Package multi-draw physical retransmission development results."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
archive = out / 'TokCom_stage14_random_retransmission_comparison.zip'
files = {
    'smoke_stage13_physical_retransmission.py': root / 'smoke_stage13_physical_retransmission.py',
    'summarize_stage14_random_trials.py': root / 'summarize_stage14_random_trials.py',
    'package_stage14.py': Path(__file__),
    'summary.json': root / 'runs/stage14_random_trials_summary.json',
    'TokCom_第十四阶段多次随机对照与重传开销_2026-10-29.md':
        out / 'TokCom_第十四阶段多次随机对照与重传开销_2026-10-29.md',
    **{f'seed{seed}.json': root / f'runs/stage14_random_trials_seed{seed}/results.json'
       for seed in range(20261029, 20261032)},
}
for seed in range(20261029, 20261032):
    case = json.loads(files[f'seed{seed}.json'].read_text(encoding='utf-8'))
    assert case['seed'] == seed and len(case['rows']) == 100
    assert case['random_trials_per_image'] == 5
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, source in files.items(): zipped.write(source, 'stage14/' + name)
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) == len(set(zipped.namelist()))
print(archive, archive.stat().st_size, 'bytes,', len(files), 'unique entries')
