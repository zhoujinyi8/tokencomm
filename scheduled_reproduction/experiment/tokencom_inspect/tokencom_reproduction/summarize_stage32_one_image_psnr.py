"""Validate the five original-image PSNR probes on one development image."""
import json
from pathlib import Path

run = Path(__file__).resolve().parent / 'runs'
rows = []
for trial in range(5):
    path = (run / 'stage31_author_metric_smoke_trial0_image0.json' if trial == 0 else
            run / f'stage32_author_psnr_trial{trial}_image0.json')
    current = json.loads(path.read_text(encoding='utf-8'))
    assert current['trial'] == trial and current['image_index'] == 0
    prior_path = (run / 'stage27_physical_reconstruction_one_image.json' if trial == 0 else
                  run / f'stage29_physical_reconstruction_trial{trial}_image0.json')
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    for now, old in zip(current['rows'], prior['rows']):
        assert now['correct_total_tokens'] == old['correct_total_tokens']
        assert abs(now['mse_to_clean_vqgan'] - old['mse_to_clean_vqgan']) < 1e-8
    rows.append({'trial': trial,
                 'oracle_psnr_raw': current['rows'][0]['original_psnr_db_author_raw'],
                 'crc_psnr_raw': current['rows'][1]['original_psnr_db_author_raw']})
result = {'scope': 'one tench development image, 5 saved AWGN 8 dB receptions',
          'mean_psnr_raw': {name: sum(row[f'{name}_psnr_raw'] for row in rows) / 5
                            for name in ('oracle', 'crc')},
          'rows': rows}
out = run / 'stage32_one_image_psnr_summary.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
