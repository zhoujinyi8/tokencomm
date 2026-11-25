"""Recompute masked-token losses from saved logits, requiring the complete paired cohort."""
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
plan = read('stage56_prior_probe_plan.json')
progress = read('stage56_prior_probe_progress.json')
lock = read('stage56_prior_probe_lock.json')
range_audit = read('stage56_range_audit.json')
range_plan = read('stage55_range_diagnostic_plan.json')
assert range_audit['complete'] and range_audit['all_checks_passed']
assert lock['script_sha256'] == sha(root/'run_stage56_prior_probe.py')
assert lock['range_audit_sha256'] == sha(run/'stage56_range_audit.json')
assert lock['plan_sha256'] == progress['plan_sha256'] == sha(run/'stage56_prior_probe_plan.json')
assert progress['lock_sha256'] == sha(run/'stage56_prior_probe_lock.json')
mapping_path = root/'evaluate_stage9_pretrained_maskgit.py'
assert sha(mapping_path) == lock['class_mapping_script_sha256']
mapping = next(node for node in ast.parse(mapping_path.read_text(encoding='utf-8')).body
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'IMAGENET_CLASS_IDS' for t in node.targets))
labels = ast.literal_eval(mapping.value)
paths = [f'{tokenization}_{mode}' for tokenization in plan['tokenizations'] for mode in plan['class_modes']]
cases = []
for path in sorted((run/'stage56_prior_probe_cases').glob('image[0-9][0-9].json')):
    case = json.loads(path.read_text(encoding='utf-8'))
    image = case['image_index']
    assert 0 <= image < 10 and path.name == f'image{image:02d}.json'
    assert sha(path) == progress['case_sha256'][path.name] and case['lock_sha256'] == progress['lock_sha256']
    assert sha(path.with_suffix('.npz')) == case['arrays_sha256']
    assert case['label'] == labels[range_plan['image_manifest'][image]['class_name']]
    source = run/f'stage56_range_cases/image{image:02d}.json'
    assert sha(source) == case['source_range_json_sha256'] == range_audit['case_sha256'][source.name]
    source_case = json.loads(source.read_text(encoding='utf-8'))
    assert sha(source.with_suffix('.npz')) == source_case['arrays_sha256']
    expected_mask = np.sort(np.random.default_rng(plan['mask_seed_base']+image).choice(256, 128, replace=False))
    assert [row['path'] for row in case['rows']] == paths
    with np.load(path.with_suffix('.npz')) as arrays, np.load(source.with_suffix('.npz')) as source_arrays:
        assert np.array_equal(arrays['mask_indices'], expected_mask)
        for tokenization in plan['tokenizations']:
            truth = arrays[f'{tokenization}_target']
            assert np.array_equal(truth, source_arrays[f'{tokenization}_tokens'][expected_mask])
            for mode in plan['class_modes']:
                name = f'{tokenization}_{mode}'
                row = next(row for row in case['rows'] if row['path'] == name)
                logits = arrays[name].astype(np.float64)
                assert logits.shape == (128, 1025) and np.isfinite(logits).all()
                maxima = logits.max(axis=1)
                logsumexp = maxima+np.log(np.exp(logits-maxima[:, None]).sum(axis=1))
                ce = float((logsumexp-logits[np.arange(128), truth]).mean())
                correct = int((logits.argmax(axis=1) == truth).sum())
                assert math.isclose(ce, row['cross_entropy'], abs_tol=2e-5)
                assert row['correct_masked_tokens'] == correct and row['masked_tokens'] == 128
                assert row['top1_accuracy'] == correct/128
    cases.append(case)
assert len(cases) == progress['complete_images']
assert len({case['image_index'] for case in cases}) == len(cases)
result = {'plan_sha256': progress['plan_sha256'], 'execution_lock_sha256': progress['lock_sha256'],
          'all_checks_passed': True, 'complete_images': len(cases), 'complete': len(cases) == 10,
          'paths': paths, 'primary_holdout_changed': False, 'multisnr_plan_changed': False,
          'selected_condition': None, 'channel_recovery_evaluated': False, 'image_quality_evaluated': False,
          'case_sha256': progress['case_sha256']}
if result['complete']:
    assert {case['image_index'] for case in cases} == set(range(10))
    means = {name: {key: sum(next(row for row in case['rows'] if row['path'] == name)[key]
                            for case in cases)/10 for key in ('cross_entropy', 'top1_accuracy')} for name in paths}
    result.update(means_over_images=means, per_image=cases,
                  training_minus_demo_by_class_mode={mode: {key: means[f'training_{mode}'][key]-means[f'demo_{mode}'][key]
                        for key in ('cross_entropy', 'top1_accuracy')} for mode in plan['class_modes']})
(run/'stage56_prior_probe_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('case_sha256', 'per_image')}, indent=2))
