"""Audit matched pipeline cases; withhold quality means until all fifty pairs exist."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from comparison_aggregation import summarize_complete_comparison

root = Path(__file__).resolve().parent; run = root/'runs'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--progress', default='stage57_comparison_progress.json')
parser.add_argument('--output', default='stage57_comparison_audit.json')
parser.add_argument('--snapshot', action='store_true')
args = parser.parse_args()
assert (run/args.progress).resolve().parent == run.resolve()
assert (run/args.output).resolve().parent == run.resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: json.loads((run/n).read_text(encoding='utf-8'))
plan = read('stage56_end_to_end_convention_plan.json')
progress = read(args.progress); lock = read('stage57_comparison_execution_lock.json')
phy_audit = read('stage57_comparison_phy_audit.json'); source_audit = read('stage57_reuse_source_audit.json')
range_audit = read('stage56_range_audit.json')
assert phy_audit['all_checks_passed'] and phy_audit['complete']
assert lock['script_sha256'] == sha(root/'run_stage57_convention_comparison.py')
assert lock['plan_sha256'] == progress['plan_sha256'] == phy_audit['plan_sha256'] == sha(run/'stage56_end_to_end_convention_plan.json')
assert progress['lock_sha256'] == sha(run/'stage57_comparison_execution_lock.json')
assert lock['phy_audit_sha256'] == sha(run/'stage57_comparison_phy_audit.json')
assert lock['range_audit_sha256'] == sha(run/'stage56_range_audit.json')
assert lock['old_recovery_audit_sha256'] == sha(run/'stage57_reuse_source_audit.json')
assert lock['old_execution_lock_sha256'] == sha(run/'stage54_development_recovery_lock.json')
assert lock['metric_lock_sha256'] == sha(run/'stage47_metrics_lock.json')
assert sha(root/'evaluate_stage9_pretrained_maskgit.py') == read('stage56_prior_probe_lock.json')['class_mapping_script_sha256']
pipelines = ['demo', 'training_code_convention']; paths = plan['paths']
cases = []; hashes = {}; reused = 0; coordinates = {name: set() for name in pipelines}
case_dir = run/'stage57_comparison_recovery'
files = ([case_dir/name for name in sorted(progress['case_sha256'])] if args.snapshot else
         sorted(case_dir.glob('*_image[0-9][0-9]_trial[0-4].json')))
for path in files:
    case = json.loads(path.read_text(encoding='utf-8'))
    pipeline, image, trial = case['pipeline'], case['image_index'], case['trial']
    assert pipeline in pipelines and 0 <= image < 10 and 0 <= trial < 5
    assert path.name == f'{pipeline}_image{image:02d}_trial{trial}.json'
    assert (image, trial) not in coordinates[pipeline]; coordinates[pipeline].add((image, trial))
    assert sha(path) == progress['case_sha256'][path.name] and case['lock_sha256'] == progress['lock_sha256']
    arrays_path = path.with_suffix('.npz'); assert sha(arrays_path) == case['arrays_sha256']
    source_json = run/f'stage56_range_cases/image{image:02d}.json'
    source = json.loads(source_json.read_text(encoding='utf-8'))
    assert sha(source_json) == range_audit['case_sha256'][source_json.name]
    assert sha(source_json.with_suffix('.npz')) == source['arrays_sha256']
    assert case['source_sha256'] == source['source_sha256'] == plan['source_image_manifest'][image]['image_sha256']
    phy_path = (run/f'stage54_development_phy/snr4_image{image:02d}_trial{trial}.npz' if pipeline == 'demo'
                else run/f'stage57_comparison_phy/image{image:02d}_trial{trial}.npz')
    assert sha(phy_path) == case['source_phy_sha256'] == phy_audit['cases'][image*5+trial]['pipelines'][pipeline]['phy_sha256']
    assert [row['path'] for row in case['rows']] == paths
    reference_case = next(item for item in range_audit['per_image'] if item['image_index'] == image)
    reference_name = 'demo_input_raw_output' if pipeline == 'demo' else 'training_range_input_rescaled_output'
    reference = next(row for row in reference_case['rows'] if row['path'] == reference_name)
    ideal = next(row for row in case['rows'] if row['path'] == 'ideal_correct_tokens')
    assert abs(ideal['psnr_raw']-reference['psnr_raw']) < 1e-4
    assert abs(ideal['lpips_normalized']-reference['lpips_normalized']) < 1e-5
    assert abs(ideal['clip_clipped']-reference['clip_clipped']) < 1e-5
    with np.load(source_json.with_suffix('.npz')) as original, np.load(phy_path) as phy, np.load(arrays_path) as arrays:
        tx = original['demo_tokens' if pipeline == 'demo' else 'training_tokens']
        rx = phy['received']; mask = phy['oracle_token_mask'].astype(bool)
        assert case['masked_tokens'] == int(mask.sum())
        assert np.array_equal(arrays['direct'], rx) and np.array_equal(arrays['ideal_correct_tokens'], tx)
        for row in case['rows']:
            ids = arrays[row['path']]
            assert ids.shape == (256,) and ids.min() >= 0 and ids.max() < 1024
            assert row['correct_tokens'] == int((ids == tx).sum())
            assert row['whole_image_ter'] == 1-row['correct_tokens']/256
            assert 0 <= row['steps'] <= 32
            assert all(math.isfinite(row[k]) for k in ('psnr_raw', 'lpips_normalized', 'clip_clipped'))
            if row['path'] in ('without_cmi', 'with_cmi'): assert np.array_equal(ids[~mask], rx[~mask])
    if case['execution_source'] == 'reused_stage54':
        assert pipeline == 'demo'; reused += 1
        old_path = run/'stage54_development_recovery'/case['reused_source_case']
        assert old_path.name == f'snr4_image{image:02d}_trial{trial}.json'
        assert sha(old_path) == case['reused_case_sha256'] == source_audit['case_sha256'][old_path.name]
        old = json.loads(old_path.read_text(encoding='utf-8'))
        assert old['rows'] == case['rows'] and old['lock_sha256'] == lock['old_execution_lock_sha256']
        assert sha(old_path.with_suffix('.npz')) == old['latent_sha256'] == case['arrays_sha256']
    else:
        assert case['execution_source'] == 'computed_stage57'
    cases.append(case); hashes[path.name] = sha(path)
assert len(cases) == progress['pipeline_receptions'] and reused == progress['reused_demo_cases']
counts = {pipeline: len(coordinates[pipeline]) for pipeline in pipelines}
assert counts == progress['counts']
pairs = coordinates['demo'] & coordinates['training_code_convention']
complete = all(counts[pipeline] == 50 for pipeline in pipelines)
result = {'plan_sha256': progress['plan_sha256'], 'execution_lock_sha256': progress['lock_sha256'],
          'all_checks_passed': True, 'snapshot_only': args.snapshot, 'counts': counts, 'pipeline_receptions': len(cases),
          'reused_demo_cases': reused, 'computed_cases': len(cases)-reused,
          'computed_cases_by_pipeline': {'demo': counts['demo']-reused,
                                       'training_code_convention': counts['training_code_convention']},
          'complete_pairs': len(pairs), 'complete_five_trial_images': sum(all((i,t) in pairs for t in range(5)) for i in range(10)),
          'complete': complete, 'quality_means_aggregated': complete, 'case_sha256': hashes,
          'primary_holdout_changed': False, 'old_multisnr_study_changed': False, 'selected_pipeline': None,
          'images_by_pipeline': {p: len({i for i,t in coordinates[p]}) for p in pipelines}}
if complete:
    assert pairs == {(i,t) for i in range(10) for t in range(5)}
    result.update(summarize_complete_comparison(cases, paths))
(run/args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('case_sha256','per_image','paired_image_differences','means_over_images','paired_mean_difference')}, indent=2))
