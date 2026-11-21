"""Check paired reconstruction identity and aggregate complete CLIP image groups."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
metrics = ['clip_author_raw_to_pil', 'clip_clipped_to_pil', 'clip_clean_vqgan_clipped_to_pil']
groups = []
saved = 0
for image in range(10):
    values = []
    for trial in range(5):
        path = run / f'stage35_clip_trial{trial}_image{image}.json'
        if not path.exists(): continue
        current = json.loads(path.read_text(encoding='utf-8'))
        prior = json.loads((run / f'stage34_lpips_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
        assert (current['trial'], current['image_index']) == (trial, image)
        assert current['class_name'] == prior['class_name']
        for now, old in zip(current['rows'], prior['rows']):
            assert now['correct_total_tokens'] == old['correct_total_tokens']
            assert abs(now['mse_to_clean_vqgan'] - old['mse_to_clean_vqgan']) < 1e-8
            assert all(math.isfinite(now[k]) and -1.00001 <= now[k] <= 1.00001 for k in metrics)
        values.append(current)
        saved += 1
    if len(values) == 5:
        groups.append({'image_index': image, 'class_name': values[0]['class_name'],
                       'means': {strategy: {metric: sum(v['rows'][j][metric] for v in values) / 5 for metric in metrics}
                                 for j, strategy in enumerate(('oracle_data_error', 'receiver_crc_failure'))}})
means = {strategy: {metric: sum(g['means'][strategy][metric] for g in groups) / len(groups) for metric in metrics}
         for strategy in ('oracle_data_error', 'receiver_crc_failure')} if groups else {}
result = {'saved_pairs': saved, 'complete_images': len(groups), 'means': means, 'per_image': groups}
(run / 'stage35_clip_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))
