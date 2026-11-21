"""Aggregate only the complete 100-image, five-trial frozen evaluation."""
import json
import math
from pathlib import Path
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs'
protocol_path=run/'stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
metrics_lock=run/'stage47_metrics_lock.json'
lock=json.loads(metrics_lock.read_text(encoding='utf-8'))
assert lock['protocol_sha256']==sha256(protocol_path)
assert lock['metric_script_sha256']==sha256(root/'evaluate_stage47_frozen_metrics.py')
paths=['direct','oracle','crc','soft_gate_0.05','soft_gate_0.1','soft_gate_0.25']
metrics=['correct_tokens','psnr_raw','lpips_normalized','clip_clipped']
rows=[]
for image in range(100):
    for trial in range(5):
        path=run/f'stage47_holdout_metrics/image{image:03d}_trial{trial}.json'
        if not path.exists():raise ValueError('Incomplete frozen evaluation: aggregate withheld')
        current=json.loads(path.read_text(encoding='utf-8'))
        assert (current['image_index'],current['trial'])==(image,trial)
        assert current['metrics_lock_sha256']==sha256(metrics_lock)
        assert current['source_recovery_sha256']==sha256(run/f'stage46_holdout_recovery/image{image:03d}_trial{trial}.npz')
        assert len(current['rows'])==6 and [r['path'] for r in current['rows']]==paths
        for row in current['rows']:
            assert all(math.isfinite(row[m]) for m in metrics)
            assert 0<=row['correct_tokens']<=256
        rows.append(current)
per_image=[]
for image in range(100):
    subset=[r for r in rows if r['image_index']==image]
    means={name:{metric:sum(next(v for v in r['rows'] if v['path']==name)[metric] for r in subset)/5
                  for metric in metrics} for name in paths}
    differences={name:{metric:means[name][metric]-means['direct'][metric] for metric in metrics}
                 for name in paths if name!='direct'}
    per_image.append({'image_index':image,'means':means,'difference_from_direct':differences})
means={name:{metric:sum(row['means'][name][metric] for row in per_image)/100 for metric in metrics} for name in paths}
differences={name:{metric:sum(row['difference_from_direct'][name][metric] for row in per_image)/100 for metric in metrics}
             for name in paths if name!='direct'}
result={'protocol_sha256':sha256(protocol_path),'complete_images':100,'complete_receptions':500,
        'metrics_lock_sha256':sha256(metrics_lock),'means_over_images':means,
        'paired_mean_differences_from_direct':differences,'per_image':per_image,
        'selected_exploratory_fraction':None,
        'caveats':protocol['limits']+['oracle path uses sender truth; gate fractions remain exploratory']}
out=run/'stage47_complete_holdout_summary.json'
if out.exists():raise ValueError('Completed evaluation summary already exists; preserve original result')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('complete frozen holdout summary saved; all exploratory paths retained')
