"""Package three physical retransmission development runs with unique names."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
archive = out / 'TokCom_stage13_physical_retransmission_dev.zip'
files = {
    'smoke_stage13_physical_retransmission.py': root / 'smoke_stage13_physical_retransmission.py',
    'summarize_stage13_physical.py': root / 'summarize_stage13_physical.py',
    'verify_stage13_initial_link.py': root / 'verify_stage13_initial_link.py',
    'package_stage13.py': Path(__file__),
    'seed20261029.json': root / 'runs/stage13_physical_retransmission_val100/results.json',
    'seed20261030.json': root / 'runs/stage13_physical_retransmission_val100_seed20261030/results.json',
    'seed20261031.json': root / 'runs/stage13_physical_retransmission_val100_seed20261031/results.json',
    'multiseed_summary.json': root / 'runs/stage13_physical_retransmission_summary.json',
    'TokCom_第十三阶段物理重传开发验证_2026-10-29.md':
        out / 'TokCom_第十三阶段物理重传开发验证_2026-10-29.md',
}
for seed in (20261029, 20261030, 20261031):
    result = json.loads(files[f'seed{seed}.json'].read_text(encoding='utf-8'))
    assert result['seed'] == seed and len(result['rows']) == 100
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
    for name, source in files.items(): zipped.write(source, 'stage13/' + name)
with ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    assert len(zipped.namelist()) == len(files) == len(set(zipped.namelist()))
print(archive, archive.stat().st_size, 'bytes,', len(files), 'unique entries')
