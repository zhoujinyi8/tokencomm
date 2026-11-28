"""Pair stage11 PSNR and stage17 ResNet18 proxy on identical development images."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
out_dir = root.parents[2] / 'outputs'
psnr_data = json.loads((root / 'runs/stage11_pretrained_reconstruction_full/results.json').read_text(encoding='utf-8'))
feature_data = json.loads((root / 'runs/stage17_resnet_feature_full/results.json').read_text(encoding='utf-8'))
assert psnr_data['images'] == feature_data['images'] == 100
assert psnr_data['data_identity'] == feature_data['data_identity']
assert psnr_data['checkpoint_sha256'] == feature_data['maskgit_checkpoint_sha256']
assert psnr_data['seed'] == feature_data['seed'] == 20261021 and psnr_data['per'] == feature_data['per'] == .3
rows = []
for a, b in zip(psnr_data['rows'], feature_data['rows']):
    assert a['source_index'] == b['source_index'] and a['image_sha256'] == b['image_sha256']
    assert a['missing_tokens'] == b['missing_tokens']
    rows.append({
        'source_index': a['source_index'], 'class': a['class'],
        'image_sha256': a['image_sha256'],
        'token_accuracy_difference': (a['token_hits']['pretrained_masked_class'] - a['token_hits']['spatial']) / a['missing_tokens'],
        'psnr_difference_db': a['psnr_to_original_db']['pretrained_masked_class'] - a['psnr_to_original_db']['spatial'],
        'feature_cosine_difference': b['feature_cosine_to_original']['pretrained_masked_class'] - b['feature_cosine_to_original']['spatial'],
        'pretrained_top1_agrees': b['top1_agreement_to_original']['pretrained_masked_class'],
        'spatial_top1_agrees': b['top1_agreement_to_original']['spatial'],
    })
x = np.array([r['psnr_difference_db'] for r in rows])
y = np.array([r['feature_cosine_difference'] for r in rows])
t = np.array([r['token_accuracy_difference'] for r in rows])
assert not np.any(x == 0) and not np.any(y == 0)
discordant = [r for r in rows if r['psnr_difference_db'] * r['feature_cosine_difference'] < 0]
quadrants = {
    'both_pretrained_better': int(((x > 0) & (y > 0)).sum()),
    'psnr_pretrained_feature_spatial': int(((x > 0) & (y < 0)).sum()),
    'psnr_spatial_feature_pretrained': int(((x < 0) & (y > 0)).sum()),
    'both_spatial_better': int(((x < 0) & (y < 0)).sum()),
}
result = {
    'scope': 'development-only paired PSNR versus frozen ResNet18 feature proxy',
    'images': 100, 'quadrants': quadrants,
    'discordant_count': len(discordant),
    'discordant_top1_agreement_status_differs': sum(
        r['pretrained_top1_agrees'] != r['spatial_top1_agrees'] for r in discordant),
    'discordant_rows': discordant,
    'pearson_correlation_of_differences': float(np.corrcoef(x, y)[0, 1]),
    'token_advantage_positive_images': int((t > 0).sum()),
    'pearson_token_advantage_vs_psnr_difference': float(np.corrcoef(t, x)[0, 1]),
    'pearson_token_advantage_vs_feature_difference': float(np.corrcoef(t, y)[0, 1]),
    'mean_psnr_difference_db': float(x.mean()),
    'mean_feature_cosine_difference': float(y.mean()),
    'note': 'A positive difference favors pretrained MaskGIT over spatial filling. '
            'Correlation is descriptive; these metrics have different units and meanings.'}
run_dir = root / 'runs/stage18_cross_metric'
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
colors = np.where(x * y < 0, '#d95f02', '#1b6ca8')
ax.scatter(x, y, c=colors, s=28, alpha=.8, edgecolors='none')
for r in discordant:
    offsets = {17: (5, -11), 34: (5, 7), 64: (-19, -14)}
    ax.annotate(str(r['source_index']),
                (r['psnr_difference_db'], r['feature_cosine_difference']),
                xytext=offsets.get(r['source_index'], (4, 4)),
                textcoords='offset points', fontsize=7)
ax.axhline(0, color='#555555', linewidth=.8)
ax.axvline(0, color='#555555', linewidth=.8)
ax.set_xlabel('PSNR difference: pretrained minus spatial (dB)')
ax.set_ylabel('ResNet18 feature cosine difference')
ax.set_title('TokCom development images: paired metric differences (n=100)')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(out_dir / 'TokCom_stage18_psnr_feature_scatter.png')
plt.close(fig)
print(json.dumps(result, indent=2))
