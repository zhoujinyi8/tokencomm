"""Validate separate development CMI files and reconcile the preselected smoke."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage53_development_cmi_grid_plan.json'
summary_path = run / 'stage53_development_cmi_grid_summary.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
summary = json.loads(summary_path.read_text(encoding='utf-8'))
metric_lock_path = run / 'stage47_metrics_lock.json'
metric_lock = json.loads(metric_lock_path.read_text(encoding='utf-8'))
assert importlib.metadata.version('lpips') == '0.1.4'
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt') == metric_lock['clip_weight_sha256']
linear = importlib.metadata.distribution('lpips').locate_file('lpips/weights/v0.1/alex.pth')
assert sha(Path(linear)) == metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth') == metric_lock['alexnet_sha256']
assert plan['script_sha256'] == sha(root / 'run_stage53_development_cmi_grid.py')
assert summary['plan_sha256'] == sha(plan_path)
assert summary['primary_holdout_changed'] is False
for name, digest in plan['input_sha256'].items():
    assert sha(root / name) == digest
with np.load(run / 'stage26_official_demo_tokens.npz') as source:
    images = json.loads(source['metadata'].item())['rows']
with np.load(run / 'stage26_official_token_link_snr8_5trials.npz') as link:
    expected_direct = float((link['transmitted'][None] == link['received']).sum()) / 50
    masks = link['oracle_token_mask'].copy()
paths = ['direct', 'without_cmi', 'with_cmi']
fields = ['correct_tokens', 'psnr_raw', 'lpips_normalized', 'clip_clipped']
cases = []
for path in sorted((run / 'stage53_development_cmi_grid').glob('image[0-9][0-9]_trial[0-4].json')):
    case = json.loads(path.read_text(encoding='utf-8'))
    image, trial = case['image_index'], case['trial']
    assert 0 <= image < 10 and 0 <= trial < 5
    assert path.name == f'image{image:02d}_trial{trial}.json'
    assert sha(path) == summary['case_sha256'][path.name]
    assert case['source_sha256'] == images[image]['image_sha256']
    assert case['plan_sha256'] == sha(plan_path)
    assert case['masked_tokens'] == int(masks[trial, image].sum())
    assert [row['path'] for row in case['rows']] == paths
    for row in case['rows']:
        assert 0 <= row['correct_tokens'] <= 256 and 0 <= row['steps'] <= 32
        assert all(math.isfinite(row[key]) for key in fields)
    cases.append(case)
assert len(cases) == summary['completed_cases']
complete = len(cases) == 50
assert summary['complete'] == complete
if complete:
    assert {(case['image_index'],case['trial']) for case in cases} == {(i,t) for i in range(10) for t in range(5)}
    assert math.isclose(summary['means_over_images']['direct']['correct_tokens'], expected_direct, abs_tol=1e-10)
    for key in fields:
        expected = summary['means_over_images']['with_cmi'][key] - summary['means_over_images']['without_cmi'][key]
        assert math.isclose(summary['difference_with_minus_without_cmi'][key], expected, abs_tol=1e-10)
else:
    assert 'means_over_images' not in summary
smoke = json.loads((run / 'stage53_development_cmi_result.json').read_text(encoding='utf-8'))
first = next((case for case in cases if case['image_index']==0 and case['trial']==0), None)
if first:
    expected = {'direct': smoke['direct_correct_tokens'], 'without_cmi': smoke['rows'][0]['correct_tokens'],
                'with_cmi': smoke['rows'][2]['correct_tokens']}
    assert all(row['correct_tokens'] == expected[row['path']] for row in first['rows'])
result = {'all_checks_passed': True, 'scope': plan['scope'], 'completed_cases':len(cases),
          'complete':complete, 'preselected_smoke_case_consistent': first is not None,
          'independent_direct_mean_correct_tokens':expected_direct,
          'source_plan_sha256':sha(plan_path), 'summary_sha256':sha(summary_path),
          'metric_weight_identities_match_primary':True, 'primary_metric_lock_sha256':sha(metric_lock_path),
          'primary_holdout_changed':False, 'primary_summary_sha256':sha(run/'stage47_complete_holdout_summary.json')}
(run/'stage53_development_cmi_grid_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
