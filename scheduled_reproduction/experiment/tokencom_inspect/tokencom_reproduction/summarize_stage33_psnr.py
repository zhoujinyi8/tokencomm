"""Validate and aggregate author-style original-image PSNR across complete images."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
rows = []
for trial in range(5):
    for image in range(10):
        if trial == 0:
            name = f'stage31_author_metric_smoke_trial0_image{image}.json'
            previous = ('stage27_physical_reconstruction_one_image.json' if image == 0 else
                        'stage27_physical_reconstruction_image1.json' if image == 1 else
                        f'stage28_physical_reconstruction_image{image}.json')
        elif image == 0:
            name = f'stage32_author_psnr_trial{trial}_image0.json'
            previous = f'stage29_physical_reconstruction_trial{trial}_image0.json'
        else:
            name = f'stage33_author_psnr_trial{trial}_image{image}.json'
            previous = f'stage30_physical_reconstruction_trial{trial}_image{image}.json'
        path = run / name
        if not path.exists(): continue
        current = json.loads(path.read_text(encoding='utf-8'))
        old = json.loads((run / previous).read_text(encoding='utf-8'))
        assert (current['trial'], current['image_index']) == (trial, image)
        assert current['class_name'] == old['class_name']
        for new_row, old_row in zip(current['rows'], old['rows']):
            assert new_row['correct_total_tokens'] == old_row['correct_total_tokens']
            assert abs(new_row['mse_to_clean_vqgan'] - old_row['mse_to_clean_vqgan']) < 1e-8
            assert new_row['output_finite']
        rows.append({'trial': trial, 'image_index': image,
                     'class_name': current['class_name'],
                     'oracle_psnr_raw': current['rows'][0]['original_psnr_db_author_raw'],
                     'crc_psnr_raw': current['rows'][1]['original_psnr_db_author_raw'],
                     'oracle_psnr_clipped': current['rows'][0]['original_psnr_db_clipped_diagnostic'],
                     'crc_psnr_clipped': current['rows'][1]['original_psnr_db_clipped_diagnostic']})
complete_images = [i for i in range(10) if sum(row['image_index'] == i for row in rows) == 5]
complete = [row for row in rows if row['image_index'] in complete_images]
per_image = []
for i in complete_images:
    subset = [row for row in complete if row['image_index'] == i]
    per_image.append({'image_index': i, 'class_name': subset[0]['class_name'],
                      **{f'{name}_mean_psnr_raw': sum(row[f'{name}_psnr_raw'] for row in subset) / 5
                         for name in ('oracle', 'crc')}})
result = {'scope': 'ten development images, five saved AWGN 8 dB receptions each; image is the aggregation unit',
          'saved_pairs': len(rows), 'complete_images': complete_images,
          'complete_pairs': len(complete),
          'mean_psnr_raw_over_complete_images': {
              name: sum(row[f'{name}_mean_psnr_raw'] for row in per_image) / len(per_image) if per_image else None
              for name in ('oracle', 'crc')},
          'mean_psnr_clipped_diagnostic': {
              name: sum(row[f'{name}_psnr_clipped'] for row in complete) / len(complete) if complete else None
              for name in ('oracle', 'crc')},
          'per_image': per_image, 'rows': rows}
out = run / 'stage33_author_psnr_10images_5trials.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('per_image', 'rows')}, indent=2))
