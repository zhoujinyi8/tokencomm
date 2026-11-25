"""Pair LPIPS diagnostics with prior reconstruction records."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
rows = []
metrics = ['lpips_author_clipped_no_normalize', 'lpips_clipped_normalize_true',
           'lpips_clean_vqgan_normalize_true']
for i in range(10):
    path = run / f'stage34_lpips_trial0_image{i}.json'
    if not path.exists(): continue
    current = json.loads(path.read_text(encoding='utf-8'))
    prior = json.loads((run / f'stage31_author_metric_smoke_trial0_image{i}.json').read_text(encoding='utf-8'))
    assert current['trial'] == 0 and current['image_index'] == i
    for now, old in zip(current['rows'], prior['rows']):
        assert now['correct_total_tokens'] == old['correct_total_tokens']
        assert abs(now['mse_to_clean_vqgan'] - old['mse_to_clean_vqgan']) < 1e-8
        assert all(math.isfinite(now[k]) for k in metrics)
        rows.append({'image_index': i, 'class_name': current['class_name'],
                     'strategy': now['mask_strategy'], **{k: now[k] for k in metrics}})
means = {strategy: {metric: sum(r[metric] for r in rows if r['strategy'] == strategy) /
                    sum(r['strategy'] == strategy for r in rows) for metric in metrics}
         for strategy in sorted({r['strategy'] for r in rows})}
result = {'images': len(rows) // 2, 'scope': 'development images, PHY trial 0, LPIPS AlexNet 0.1.4',
          'means': means, 'rows': rows}
(run / 'stage34_lpips_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
