"""Paired analysis of full stage17 development-set feature diagnostics."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
path = root / 'runs/stage17_resnet_feature_full/results.json'
data = json.loads(path.read_text(encoding='utf-8'))
rows = data['rows']
assert len(rows) == data['images'] == 100
assert [r['source_index'] for r in rows] == list(range(100))
names = ('clean_vqgan', 'pretrained_masked_class', 'spatial')
cos = {name: np.array([r['feature_cosine_to_original'][name] for r in rows]) for name in names}
agreement = {name: np.array([r['top1_agreement_to_original'][name] for r in rows]) for name in names}
for name in names:
    assert abs(float(cos[name].mean()) - data['mean_feature_cosine_to_original'][name]) < 1e-12
    assert abs(float(agreement[name].mean()) - data['top1_agreement_to_original'][name]) < 1e-12
delta = cos['pretrained_masked_class'] - cos['spatial']
draw = np.random.default_rng(20261101).integers(0, 100, size=(10000, 100))
interval = np.quantile(delta[draw].mean(1), [.025, .975])
result = {
    'scope': 'development-only paired ResNet18 feature proxy; no test images',
    'images': 100,
    'mean_cosine_to_original': {name: float(cos[name].mean()) for name in names},
    'top1_agreement_to_original': {name: float(agreement[name].mean()) for name in names},
    'pretrained_minus_spatial_mean_cosine': float(delta.mean()),
    'pretrained_higher_images': int((delta > 0).sum()),
    'pretrained_lower_images': int((delta < 0).sum()),
    'paired_image_bootstrap_95pct_difference': [float(x) for x in interval],
    'pretrained_agrees_and_spatial_disagrees': int(
        (agreement['pretrained_masked_class'] & ~agreement['spatial']).sum()),
    'spatial_agrees_and_pretrained_disagrees': int(
        (agreement['spatial'] & ~agreement['pretrained_masked_class']).sum()),
    'caution': 'Frozen ImageNet-pretrained ResNet18 on Imagenette development data. '
               'Neither CLIP/LPIPS nor an unseen-pretraining-data benchmark.'}
out = root / 'runs/stage17_resnet_feature_full/analysis.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
