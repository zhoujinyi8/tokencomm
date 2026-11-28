"""Summarize feedback retry sensitivity without treating repeated seeds as new images."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
paths = [root / 'runs/stage16_feedback_retry_summary.json'] + [
    root / f'runs/stage16_feedback_retry_seed{s}.json' for s in range(20261030, 20261035)]
cases = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
assert [x['feedback_seed'] for x in cases] == list(range(20261029, 20261035))
assert all(x['max_feedback_attempts'] == 2 and x['missing_tokens_across_seeds'] == 24320
           for x in cases)
rows = []
for case in cases:
    low = case['low']
    random = case['random_five_trial_mean']
    rows.append({'feedback_seed': case['feedback_seed'],
                 'low_accuracy': low['accuracy'], 'random_accuracy': random['accuracy'],
                 'difference': case['accuracy_difference'],
                 'low_added_symbols_per_image': low['added_symbols_per_request'],
                 'random_added_symbols_per_image': random['added_symbols_per_request'],
                 'low_feedback_crc_passes': low['feedback_crc_passes'],
                 'random_feedback_crc_passes': random['feedback_crc_passes']})
result = {
    'scope': 'six feedback noise draws on identical development images and cached data outcomes',
    'feedback_seeds': [r['feedback_seed'] for r in rows],
    'difference_range': [min(r['difference'] for r in rows), max(r['difference'] for r in rows)],
    'low_accuracy_range': [min(r['low_accuracy'] for r in rows), max(r['low_accuracy'] for r in rows)],
    'random_accuracy_range': [min(r['random_accuracy'] for r in rows), max(r['random_accuracy'] for r in rows)],
    'rows': rows,
    'caution': 'Repeated feedback seeds reuse all images and physical data retransmission outcomes.'}
out = root / 'runs/stage16_feedback_retry_sensitivity.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
