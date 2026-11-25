"""Audit per-case metric identities and coverage without interim aggregation."""
import json
import math
from pathlib import Path
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs'
lock_path=run/'stage47_metrics_lock.json'
lock=json.loads(lock_path.read_text(encoding='utf-8'))
assert lock['metric_script_sha256']==sha256(root/'evaluate_stage47_frozen_metrics.py')
assert lock['protocol_sha256']==sha256(run/'stage44_frozen_protocol.json')
paths=['direct','oracle','crc','soft_gate_0.05','soft_gate_0.1','soft_gate_0.25']
cases=[]
for path in sorted((run/'stage47_holdout_metrics').glob('image[0-9][0-9][0-9]_trial[0-4].json')):
    image=int(path.stem[5:8]);trial=int(path.stem[-1])
    current=json.loads(path.read_text(encoding='utf-8'))
    assert (current['image_index'],current['trial'])==(image,trial)
    assert current['metrics_lock_sha256']==sha256(lock_path)
    recovery=run/f'stage46_holdout_recovery/image{image:03d}_trial{trial}.npz'
    assert current['source_recovery_sha256']==sha256(recovery)
    assert len(current['rows'])==6 and [r['path'] for r in current['rows']]==paths
    for row in current['rows']:
        assert 0<=row['correct_tokens']<=256
        assert all(math.isfinite(row[key]) for key in ('psnr_raw','lpips_normalized','clip_clipped'))
    cases.append({'image_index':image,'trial':trial,'sha256':sha256(path)})
result={'metric_receptions_saved':len(cases),'paths_per_reception':6,
        'complete_five_trial_images':sum(sum(c['image_index']==i for c in cases)==5 for i in range(100)),
        'expected_receptions':500,'all_checks_passed':True,'performance_aggregated':False,'cases':cases}
(run/'stage47_metric_file_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
