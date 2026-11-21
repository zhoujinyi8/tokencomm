"""Validate saved paired reconstructions and summarize complete trial groups."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root / 'runs'
physical = json.loads(np.load(run / 'stage26_official_token_link_snr8_5trials.npz')['metadata'].item())
cache = np.load(run / 'stage26_official_token_link_snr8_5trials.npz')
rows = []
for trial in range(5):
    for image in range(10):
        if trial == 0:
            filename = ('stage27_physical_reconstruction_one_image.json' if image == 0 else
                        'stage27_physical_reconstruction_image1.json' if image == 1 else
                        f'stage28_physical_reconstruction_image{image}.json')
        elif image == 0:
            filename = f'stage29_physical_reconstruction_trial{trial}_image0.json'
        else:
            filename = f'stage30_physical_reconstruction_trial{trial}_image{image}.json'
        path = run / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        assert (data['trial'], data['image_index'], data['snr_db']) == (trial, image, 8.)
        phy = physical['rows'][trial * 10 + image]
        assert data['class_name'] == phy['class_name']
        oracle, crc = data['rows']
        for name, item, mask, expected in [
            ('oracle_data_error', oracle, 'oracle_token_mask', phy['oracle_bad_packets']),
            ('receiver_crc_failure', crc, 'crc_token_mask', phy['crc_bad_packets'])]:
            masked = int(cache[mask][trial, image].sum())
            assert item['mask_strategy'] == name
            assert item['masked_tokens'] == masked == expected * 4
            assert item['correct_unmasked_received_tokens'] == 256 - masked
            assert item['output_finite']
            assert 0 <= item['correct_total_tokens'] <= 256
        rows.append({'trial': trial, 'image': image, 'class_name': data['class_name'],
                     'oracle_correct': oracle['correct_total_tokens'],
                     'crc_correct': crc['correct_total_tokens'],
                     'oracle_mse': oracle['mse_to_clean_vqgan'],
                     'crc_mse': crc['mse_to_clean_vqgan']})
complete_images = [image for image in range(10)
                   if sum(row['image'] == image for row in rows) == 5]
complete = [row for row in rows if row['image'] in complete_images]
summary = {'scope': 'stage5 development images, saved AWGN 8 dB PHY receptions; repeated trials are paired within image',
           'saved_pairs': len(rows), 'complete_images': complete_images,
           'complete_pairs': len(complete),
           'mean_correct_per_image_trial': {
               name: sum(row[f'{name}_correct'] for row in complete) / len(complete) if complete else None
               for name in ('oracle', 'crc')},
           'mean_mse_per_image_trial': {
               name: sum(row[f'{name}_mse'] for row in complete) / len(complete) if complete else None
               for name in ('oracle', 'crc')},
           'rows': rows}
out = run / 'stage30_multitrial_progress.json'
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
