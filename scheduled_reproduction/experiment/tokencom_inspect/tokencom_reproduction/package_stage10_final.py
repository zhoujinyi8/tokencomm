"""Package the frozen stage10 test result with the prior validation materials."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / 'outputs'
OLD = OUT / 'TokCom_stage10_validation_holdout_increment.zip'
NEW = OUT / 'TokCom_stage10_frozen_holdout_final.zip'
REPORT = OUT / 'TokCom_第十阶段独立留出测试最终结果_2026-10-25.md'
RUN = ROOT / 'runs' / 'stage10_frozen_holdout'

status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
summary = json.loads((RUN / 'summary_final.json').read_text(encoding='utf-8'))
assert status['complete'] and status['cases_completed'] == status['cases_expected'] == 15
assert summary['complete'] and len(summary['summaries']) == 3

prefix = 'stage10_final/'
replacements = {
    'stage10_holdout_progress.json': RUN / 'progress.json',
    'stage10_holdout_status.json': RUN / 'status.json',
    'stage10_holdout_summary_final.json': RUN / 'summary_final.json',
    REPORT.name: REPORT,
}
with ZipFile(OLD) as old, ZipFile(NEW, 'w', compression=ZIP_DEFLATED) as new:
    for info in old.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name in {'stage10_holdout_progress.json', 'stage10_holdout_status.json',
                    'stage10_holdout_summary_partial.json'} or name.endswith('_2026-10-25.md'):
            continue
        rel = '/'.join(info.filename.split('/')[1:])
        new.writestr(prefix + rel, old.read(info.filename))
    for name, source in replacements.items():
        new.write(source, prefix + name)

with ZipFile(NEW) as zipped:
    assert zipped.testzip() is None
    names = zipped.namelist()
    assert sum(name.lower().endswith(('.jpg', '.jpeg', '.png')) for name in names) == 100
    assert all(prefix + name in names for name in replacements)
print(NEW)
print(f'{NEW.stat().st_size} bytes; {len(names)} entries; complete {status["cases_completed"]}/15')
