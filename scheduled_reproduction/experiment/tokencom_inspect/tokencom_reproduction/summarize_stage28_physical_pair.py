"""Pair ten author reconstructions from the same first PHY trial."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
physical = json.loads(__import__('numpy').load(run / 'stage26_official_token_link_snr8_5trials.npz')['metadata'].item())
files = [run / 'stage27_physical_reconstruction_one_image.json',
         run / 'stage27_physical_reconstruction_image1.json']
files += [run / f'stage28_physical_reconstruction_image{i}.json' for i in range(2, 10)]
rows = []
for index, path in enumerate(files):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['trial'] == 0 and data['image_index'] == index and data['snr_db'] == 8.
    oracle, crc = data['rows']
    assert oracle['mask_strategy'] == 'oracle_data_error'
    assert crc['mask_strategy'] == 'receiver_crc_failure'
    phy = physical['rows'][index]
    assert data['class_name'] == phy['class_name']
    assert oracle['masked_tokens'] == phy['oracle_bad_packets'] * 4
    assert crc['masked_tokens'] == phy['crc_bad_packets'] * 4
    assert oracle['correct_unmasked_received_tokens'] == 256 - oracle['masked_tokens']
    assert crc['correct_unmasked_received_tokens'] == 256 - crc['masked_tokens']
    rows.append({'image_index': index, 'class_name': data['class_name'],
                 'oracle_correct_tokens': oracle['correct_total_tokens'],
                 'crc_correct_tokens': crc['correct_total_tokens'],
                 'oracle_mse': oracle['mse_to_clean_vqgan'],
                 'crc_mse': crc['mse_to_clean_vqgan'],
                 'oracle_masked_tokens': oracle['masked_tokens'],
                 'crc_masked_tokens': crc['masked_tokens']})
result = {
    'scope': '10 stage5 dev images, one common Sionna reception per image; fixed true class and sampling seed; no new blind test',
    'images': len(rows), 'snr_db': 8., 'trial': 0,
    'sum_correct_tokens': {name: sum(row[f'{name}_correct_tokens'] for row in rows)
                           for name in ('oracle', 'crc')},
    'mean_mse_to_clean_vqgan': {name: sum(row[f'{name}_mse'] for row in rows) / len(rows)
                                for name in ('oracle', 'crc')},
    'oracle_higher_token_images': sum(r['oracle_correct_tokens'] > r['crc_correct_tokens'] for r in rows),
    'crc_higher_token_images': sum(r['crc_correct_tokens'] > r['oracle_correct_tokens'] for r in rows),
    'oracle_lower_mse_images': sum(r['oracle_mse'] < r['crc_mse'] for r in rows),
    'crc_lower_mse_images': sum(r['crc_mse'] < r['oracle_mse'] for r in rows),
    'rows': rows,
}
out = run / 'stage28_physical_reconstruction_pair_summary.json'
if out.exists(): raise ValueError('Output exists')
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
