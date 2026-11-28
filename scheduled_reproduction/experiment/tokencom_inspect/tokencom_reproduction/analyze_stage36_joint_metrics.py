"""Pair all four metrics within image and PHY trial before averaging."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
records = []
for trial in range(5):
    for image in range(10):
        lp = json.loads((run / f'stage34_lpips_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
        cl = json.loads((run / f'stage35_clip_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
        assert (lp['trial'], lp['image_index'], lp['class_name']) == (cl['trial'], cl['image_index'], cl['class_name'])
        values = {}
        for index, label in enumerate(('oracle', 'crc')):
            a, b = lp['rows'][index], cl['rows'][index]
            assert a['correct_total_tokens'] == b['correct_total_tokens']
            assert abs(a['mse_to_clean_vqgan'] - b['mse_to_clean_vqgan']) < 1e-8
            values[label] = {'tokens': a['correct_total_tokens'],
                             'psnr_raw': a['original_psnr_db_author_raw'],
                             'lpips_normalized': a['lpips_clipped_normalize_true'],
                             'clip_clipped': b['clip_clipped_to_pil'],
                             'lpips_author': a['lpips_author_clipped_no_normalize'],
                             'clip_author': b['clip_author_raw_to_pil']}
        records.append({'trial': trial, 'image_index': image, 'class_name': lp['class_name'], **values})
metrics = list(records[0]['oracle'])
per_image = []
for image in range(10):
    selected = [r for r in records if r['image_index'] == image]
    assert len(selected) == 5
    values = {label: {metric: sum(r[label][metric] for r in selected) / 5 for metric in metrics}
              for label in ('oracle', 'crc')}
    advantage = {metric: (values['crc'][metric] - values['oracle'][metric] if metric.startswith('lpips')
                          else values['oracle'][metric] - values['crc'][metric]) for metric in metrics}
    per_image.append({'image_index': image, 'class_name': selected[0]['class_name'],
                      **values, 'oracle_favorable_difference': advantage})
counts = {metric: {'oracle': sum(r['oracle_favorable_difference'][metric] > 1e-8 for r in per_image),
                   'crc': sum(r['oracle_favorable_difference'][metric] < -1e-8 for r in per_image),
                   'tie': sum(abs(r['oracle_favorable_difference'][metric]) <= 1e-8 for r in per_image)}
          for metric in metrics}
disagreements = [r['class_name'] for r in per_image
                 if r['oracle_favorable_difference']['tokens'] * r['oracle_favorable_difference']['clip_clipped'] < 0]
result = {'scope': '10 development images; five paired PHY trials averaged within image; descriptive only',
          'pairs': 50, 'images': 10, 'metric_direction_counts_over_images': counts,
          'token_clip_disagreement_images': disagreements, 'per_image': per_image, 'paired_records': records}
(run / 'stage36_joint_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('per_image', 'paired_records')}, indent=2))
