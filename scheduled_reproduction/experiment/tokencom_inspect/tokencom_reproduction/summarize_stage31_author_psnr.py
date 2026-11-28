"""Summarize available original-image PSNR probes without treating images as blind test."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
files = [root / f'runs/stage31_author_metric_smoke_trial0_image{i}.json' for i in range(10)]
rows = []
for i, path in enumerate(files):
    if not path.exists(): continue
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['trial'] == 0 and data['image_index'] == i
    oracle, crc = data['rows']
    assert oracle['mask_strategy'] == 'oracle_data_error'
    assert crc['mask_strategy'] == 'receiver_crc_failure'
    prior_path = (root / 'runs/stage27_physical_reconstruction_one_image.json' if i == 0 else
                  root / 'runs/stage27_physical_reconstruction_image1.json' if i == 1 else
                  root / f'runs/stage28_physical_reconstruction_image{i}.json')
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    for current, old in zip(data['rows'], prior['rows']):
        assert current['correct_total_tokens'] == old['correct_total_tokens']
        assert abs(current['mse_to_clean_vqgan'] - old['mse_to_clean_vqgan']) < 1e-8
    rows.append({'image_index': i, 'class_name': data['class_name'],
                 'oracle_psnr_raw': oracle['original_psnr_db_author_raw'],
                 'crc_psnr_raw': crc['original_psnr_db_author_raw'],
                 'oracle_psnr_clipped': oracle['original_psnr_db_clipped_diagnostic'],
                 'crc_psnr_clipped': crc['original_psnr_db_clipped_diagnostic'],
                 'oracle_outside_fraction': oracle['output_outside_0_1_fraction'],
                 'crc_outside_fraction': crc['output_outside_0_1_fraction']})
result = {'scope': 'same ten development images, PHY trial 0; original-image PSNR uses raw author decoder output',
          'images_complete': len(rows),
          'mean_psnr_raw': {name: sum(row[f'{name}_psnr_raw'] for row in rows) / len(rows) if rows else None
                            for name in ('oracle', 'crc')},
          'mean_psnr_clipped_diagnostic': {
              name: sum(row[f'{name}_psnr_clipped'] for row in rows) / len(rows) if rows else None
              for name in ('oracle', 'crc')},
          'rows': rows}
out = root / 'runs/stage31_author_psnr_summary.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
