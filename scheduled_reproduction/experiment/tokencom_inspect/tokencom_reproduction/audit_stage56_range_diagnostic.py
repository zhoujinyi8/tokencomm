"""Audit saved convention arrays and aggregate only all ten preregistered images."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
plan = read('stage55_range_diagnostic_plan.json')
progress = read('stage56_range_progress.json')
lock = read('stage56_range_execution_lock.json')
assert progress['plan_sha256'] == lock['plan_sha256'] == sha(run/'stage55_range_diagnostic_plan.json')
assert lock['script_sha256'] == sha(root/'run_stage56_range_diagnostic.py')
assert progress['lock_sha256'] == sha(run/'stage56_range_execution_lock.json')
assert lock['metric_lock_sha256'] == sha(run/'stage47_metrics_lock.json')
paths = [condition['path'] for condition in plan['conditions']]
fields = ['psnr_raw', 'lpips_normalized', 'clip_clipped']
cases = []
data = root/'data/imagenette_stage5'
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as old:
    legacy_tokens = old['transmitted'].copy()
for path in sorted((run/'stage56_range_cases').glob('image[0-9][0-9].json')):
    case = json.loads(path.read_text(encoding='utf-8'))
    image = case['image_index']
    assert 0 <= image < 10 and path.name == f'image{image:02d}.json'
    assert sha(path) == progress['case_sha256'][path.name]
    assert case['lock_sha256'] == progress['lock_sha256']
    assert case['source_index'] == plan['selected_indices'][image]
    source = (data/case['source_image']).resolve()
    assert source.is_relative_to(data.resolve())
    assert sha(source) == case['source_sha256'] == plan['image_manifest'][image]['image_sha256']
    arrays_path = path.with_suffix('.npz')
    assert sha(arrays_path) == case['arrays_sha256']
    assert [row['path'] for row in case['rows']] == paths
    with np.load(arrays_path) as arrays:
        original = arrays['original']
        assert original.shape == (1, 3, 256, 256) and original.dtype == np.float32
        with Image.open(source) as opened:
            expected = np.array(opened.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR),
                                dtype=np.float32).transpose(2, 0, 1)[None]/np.float32(255)
        assert np.array_equal(original, expected)
        assert np.array_equal(arrays['demo_tokens'], legacy_tokens[image])
        for name in ('demo_tokens', 'training_tokens'):
            assert arrays[name].shape == (256,) and arrays[name].min() >= 0 and arrays[name].max() < 1024
        assert case['changed_tokens'] == int((arrays['demo_tokens'] != arrays['training_tokens']).sum())
        outputs = [arrays['demo_raw'], (arrays['demo_raw']+1)/2,
                   arrays['training_raw'], (arrays['training_raw']+1)/2]
        for row, output in zip(case['rows'], outputs):
            assert output.shape == original.shape and np.isfinite(output).all()
            mse = float(np.mean((output-original)**2, dtype=np.float64))
            assert math.isclose(mse, row['mse_unclipped'], rel_tol=2e-6)
            assert abs(-10*math.log10(mse)-row['psnr_raw']) < 3e-5
            assert all(math.isfinite(row[key]) for key in fields)
    cases.append(case)
assert len(cases) == progress['completed_images']
complete = len(cases) == 10
assert len({case['image_index'] for case in cases}) == len(cases)
result = {'plan_sha256': progress['plan_sha256'], 'execution_lock_sha256': progress['lock_sha256'],
          'all_checks_passed': True, 'complete': complete, 'complete_images': len(cases),
          'condition_paths': paths, 'primary_holdout_changed': False, 'multisnr_plan_changed': False,
          'maskgit_evaluated': False, 'channel_evaluated': False,
          'case_sha256': {f"image{case['image_index']:02d}.json": progress['case_sha256'][f"image{case['image_index']:02d}.json"] for case in cases},
          'selected_condition': None, 'actual_checkpoint_training_provenance_established': False}
if complete:
    assert {case['image_index'] for case in cases} == set(range(10))
    means = {name: {key: sum(next(row for row in case['rows'] if row['path'] == name)[key]
                                 for case in cases)/10 for key in fields} for name in paths}
    result.update(means_over_images=means,
                  difference_from_demo_raw={name: {key: means[name][key]-means[paths[0]][key] for key in fields} for name in paths},
                  changed_token_ids=sum(case['changed_tokens'] for case in cases), total_token_ids=2560,
                  changed_token_fraction=sum(case['changed_tokens'] for case in cases)/2560,
                  token_change_definition='identity difference between two encodings of the same image, not channel TER',
                  per_image=[{key: value for key, value in case.items() if key not in ('source_image',)} for case in cases])
(run/'stage56_range_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('case_sha256', 'per_image')}, indent=2))
