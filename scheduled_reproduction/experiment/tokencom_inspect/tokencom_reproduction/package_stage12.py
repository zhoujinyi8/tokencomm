"""Package development-only retransmission priority diagnostics."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parents[2] / 'outputs'
REPORT = OUTPUT / 'TokCom_第十二阶段重传优先级开发验证_2026-10-28.md'
ARCHIVE = OUTPUT / 'TokCom_stage12_retransmission_priority_dev.zip'
CASES = {
    'per01_budget04.json': ROOT / 'runs/stage12_retransmission_priority_val_per01/results.json',
    'per03_budget04.json': ROOT / 'runs/stage12_retransmission_priority_val/results.json',
    'per05_budget04.json': ROOT / 'runs/stage12_retransmission_priority_val_per05/results.json',
    'per03_budget02.json': ROOT / 'runs/stage12_retransmission_priority_val_budget02/results.json',
    'per03_budget08.json': ROOT / 'runs/stage12_retransmission_priority_val_budget08/results.json',
    **{f'per03_seed{seed}_budget04.json':
       ROOT / f'runs/stage12_retransmission_priority_val_seed{seed}/results.json'
       for seed in range(20261022, 20261026)},
}
for name, path in CASES.items():
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['images'] == len(result['rows']) == 100, name
    assert result['budget_packets_per_image'] == int(name[-7:-5]), name
with ZipFile(ARCHIVE, 'w', compression=ZIP_DEFLATED) as zipped:
    zipped.write(REPORT, f'stage12/{REPORT.name}')
    zipped.write(ROOT / 'diagnose_stage12_retransmission_priority.py',
                 'stage12/diagnose_stage12_retransmission_priority.py')
    zipped.write(Path(__file__), 'stage12/package_stage12.py')
    zipped.write(ROOT / 'summarize_stage12_multiseed.py',
                 'stage12/summarize_stage12_multiseed.py')
    zipped.write(ROOT / 'runs/stage12_retransmission_priority_multiseed_summary.json',
                 'stage12/multiseed_summary.json')
    for name, path in CASES.items(): zipped.write(path, f'stage12/{name}')
with ZipFile(ARCHIVE) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(CASES) + 5
print(ARCHIVE)
print(f'{ARCHIVE.stat().st_size} bytes, {len(CASES)} result cases')
