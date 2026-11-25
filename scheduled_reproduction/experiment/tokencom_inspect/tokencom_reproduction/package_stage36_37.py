"""Package joint metrics and the direct-decoding baseline without large weights."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
outputs=root.parents[2]/'outputs'
names=['analyze_stage36_joint_metrics.py','report_stage36_joint_metrics.py',
       'evaluate_stage37_direct_baseline.py','summarize_stage37_baseline.py',
       'runs/stage36_joint_metrics.json','runs/stage37_direct_baseline.json',
       'runs/stage37_direct_comparison.json','runs/stage26_official_demo_tokens.npz',
       'runs/stage26_official_token_link_snr8_5trials.npz']
items=[(name,root) for name in names]
items += [(name,outputs) for name in ['TokCom_第三十六阶段多指标配对分析_2026-11-19.md',
                                     'TokCom_第三十七阶段直接接收解码基线_2026-11-19.md']]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage36_37_joint_metrics_direct_baseline.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,base in items: archive.write(base/name,name)
    archive.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest()==h for name,h in manifest.items())
print(target,target.stat().st_size)
