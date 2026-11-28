"""Aggregate complete five-trial LPIPS image groups and check prior outputs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
metrics = ['lpips_author_clipped_no_normalize', 'lpips_clipped_normalize_true', 'lpips_clean_vqgan_normalize_true']
groups = []
saved = 0
for image in range(10):
    values = []
    for trial in range(5):
        path = run / f'stage34_lpips_trial{trial}_image{image}.json'
        if not path.exists(): continue
        current = json.loads(path.read_text(encoding='utf-8'))
        prior_name = (f'stage31_author_metric_smoke_trial0_image{image}.json' if trial == 0 else
                      f'stage32_author_psnr_trial{trial}_image0.json' if image == 0 else
                      f'stage33_author_psnr_trial{trial}_image{image}.json')
        prior = json.loads((run / prior_name).read_text(encoding='utf-8'))
        assert (current['trial'], current['image_index']) == (trial, image)
        for now, old in zip(current['rows'], prior['rows']):
            assert now['correct_total_tokens'] == old['correct_total_tokens']
            assert abs(now['mse_to_clean_vqgan'] - old['mse_to_clean_vqgan']) < 1e-8
        values.append(current)
        saved += 1
    if len(values) == 5:
        groups.append({'image_index': image, 'class_name': values[0]['class_name'],
                       'means': {strategy: {metric: sum(v['rows'][j][metric] for v in values) / 5 for metric in metrics}
                                 for j, strategy in enumerate(('oracle_data_error', 'receiver_crc_failure'))}})
means = {strategy: {metric: sum(g['means'][strategy][metric] for g in groups) / len(groups) for metric in metrics}
         for strategy in ('oracle_data_error', 'receiver_crc_failure')} if groups else {}
result = {'saved_pairs': saved, 'complete_images': len(groups), 'means': means, 'per_image': groups}
(run / 'stage34_lpips_five_trials_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))
