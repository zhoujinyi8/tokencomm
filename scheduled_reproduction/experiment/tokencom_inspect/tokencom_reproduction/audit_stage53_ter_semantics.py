"""Post hoc clarification of the author return value versus full residual TER."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
summary_path = run / 'stage47_complete_holdout_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
assert summary['complete_receptions'] == 500 and summary['complete_images'] == 100
progress = json.loads((run / 'stage46_recovery_progress.json').read_text(encoding='utf-8'))
assert progress['completed_recovery_cases'] == 500
expected = {(case['image_index'], case['trial']): case['sha256'] for case in progress['cases']}
interface = json.loads((run / 'stage53_cmi_interface_audit.json').read_text(encoding='utf-8'))
assert all(example['returned_TER_af'] == 0 and example['actual_whole_image_residual_TER'] == 1
           for example in interface['return_metric_examples'].values())
counts = {name: dict(masked_tokens=0, originally_correct_masked=0, correct_masked_after=0,
                     repaired_original_errors=0, damaged_originally_correct=0, residual_wrong_tokens=0)
          for name in ('oracle', 'crc')}
before_wrong = 0
for image in range(100):
    with np.load(run / f'stage45_holdout_tokens/image{image:03d}.npz') as source:
        tx = source['tokens'].copy()
    for trial in range(5):
        path = run / f'stage46_holdout_recovery/image{image:03d}_trial{trial}.npz'
        assert sha(path) == expected[image, trial]
        with np.load(path) as recovery, np.load(run / f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz') as phy:
            rx = recovery['direct']
            wrong = rx != tx
            before_wrong += int(wrong.sum())
            for name in counts:
                mask = phy[f'{name}_token_mask'].astype(bool)
                final = recovery[name]
                assert np.array_equal(final[~mask], rx[~mask])
                after_correct = final == tx
                c = counts[name]
                c['masked_tokens'] += int(mask.sum())
                c['originally_correct_masked'] += int((mask & ~wrong).sum())
                c['correct_masked_after'] += int((mask & after_correct).sum())
                c['repaired_original_errors'] += int((wrong & after_correct).sum())
                c['damaged_originally_correct'] += int((~wrong & ~after_correct).sum())
                c['residual_wrong_tokens'] += int((~after_correct).sum())
total_tokens = 500 * 256
for name, c in counts.items():
    assert c['residual_wrong_tokens'] == before_wrong - c['repaired_original_errors'] + c['damaged_originally_correct']
    c['source_return_formula_recomputed_from_saved_codes'] = c['correct_masked_after'] / total_tokens
    c['actual_whole_image_residual_TER'] = c['residual_wrong_tokens'] / total_tokens
    expected_correct = (total_tokens - c['residual_wrong_tokens']) / 500
    assert math.isclose(summary['means_over_images'][name]['correct_tokens'], expected_correct, abs_tol=1e-10)
phy_result = json.loads((run / 'stage53_payload_tce_result.json').read_text(encoding='utf-8'))
assert before_wrong == phy_result['totals']['wrong_received_tokens']
result = {'kind': 'post hoc descriptive definition audit after full frozen summary; not a new primary outcome',
          'original_primary_protocol_changed': False,
          'complete_receptions': 500, 'total_tokens': total_tokens,
          'direct_wrong_tokens': before_wrong, 'direct_whole_image_TER': before_wrong / total_tokens,
          'paths': counts, 'all_accounting_checks_passed': True,
          'author_return_definition': 'correct masked predictions / all image tokens',
          'source_return_values_are_recomputed': True,
          'author_source_sha256': interface['author_source_sha256'],
          'summary_sha256': sha(summary_path), 'script_sha256': sha(Path(__file__)),
          'caveat': 'Audits the pinned public code return definition; does not establish how published paper curves were calculated'}
out = run / 'stage53_ter_definition_audit.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
