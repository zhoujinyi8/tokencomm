"""Reconcile the complete frozen summary with independently audited PHY counts."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
summary_path = run / 'stage47_complete_holdout_summary.json'
metric_audit_path = run / 'stage47_metric_file_audit.json'
phy_path = run / 'stage53_payload_tce_result.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
audit = json.loads(metric_audit_path.read_text(encoding='utf-8'))
phy = json.loads(phy_path.read_text(encoding='utf-8'))
assert audit['all_checks_passed'] and audit['metric_receptions_saved'] == 500
assert audit['complete_five_trial_images'] == summary['complete_images'] == 100
assert summary['complete_receptions'] == phy['complete_receptions'] == 500
assert summary['protocol_sha256'] == phy['protocol_sha256'] == sha(run / 'stage44_frozen_protocol.json')
assert summary['metrics_lock_sha256'] == sha(run / 'stage47_metrics_lock.json')
assert summary['selected_exploratory_fraction'] is None
paths = ['direct', 'oracle', 'crc', 'soft_gate_0.05', 'soft_gate_0.1', 'soft_gate_0.25']
fields = ['correct_tokens', 'psnr_raw', 'lpips_normalized', 'clip_clipped']
assert set(summary['means_over_images']) == set(paths)
assert {row['image_index'] for row in summary['per_image']} == set(range(100))
assert len(summary['per_image']) == 100
for row in summary['per_image']:
    assert set(row['means']) == set(paths)
    for name in paths:
        assert all(math.isfinite(row['means'][name][key]) for key in fields)
        assert 0 <= row['means'][name]['correct_tokens'] <= 256
for name in paths:
    assert all(math.isfinite(summary['means_over_images'][name][key]) for key in fields)
    if name == 'direct':
        continue
    for key in fields:
        expected = summary['means_over_images'][name][key] - summary['means_over_images']['direct'][key]
        actual = summary['paired_mean_differences_from_direct'][name][key]
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
expected_direct = 256 - phy['totals']['wrong_received_tokens'] / 500
assert math.isclose(summary['means_over_images']['direct']['correct_tokens'], expected_direct,
                    rel_tol=0, abs_tol=1e-10)
result = {'all_checks_passed': True, 'complete_images': 100, 'complete_receptions': 500,
          'paths': paths, 'selected_exploratory_fraction': None,
          'direct_correct_tokens_from_independent_phy_counts': expected_direct,
          'paired_difference_arithmetic_consistent': True, 'per_image_coverage_complete': True,
          'summary_sha256': sha(summary_path), 'metric_audit_sha256': sha(metric_audit_path),
          'supplemental_phy_result_sha256': sha(phy_path),
          'original_summary_entrypoint_sha256': sha(root / 'summarize_stage47_complete_holdout.py'),
          'note': 'Integrity and cross-artifact reconciliation; no selection or extra performance outcome'}
out = run / 'stage53_complete_summary_audit.json'
assert not out.exists(), 'Preserve an existing completed summary audit'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
