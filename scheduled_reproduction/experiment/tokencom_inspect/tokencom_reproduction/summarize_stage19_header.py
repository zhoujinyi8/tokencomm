"""Summarize coded 16-bit feedback header sensitivity on the development set."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
rows = []
for seed in range(20261029, 20261035):
    no_path = run / ('stage16_feedback_retry_summary.json' if seed == 20261029 else
                     f'stage16_feedback_retry_seed{seed}.json')
    bare = json.loads(no_path.read_text(encoding='utf-8'))
    header = json.loads((run / f'stage19_header16_seed{seed}.json').read_text(encoding='utf-8'))
    assert bare['feedback_seed'] == header['feedback_seed'] == seed
    assert header['header_bits'] == 16 and header['feedback_symbols_per_attempt'] == 31
    assert bare['max_feedback_attempts'] == header['max_feedback_attempts'] == 2
    assert bare['missing_tokens_across_seeds'] == header['missing_tokens_across_seeds'] == 24320
    rows.append({
        'feedback_seed': seed,
        'no_header_low_accuracy': bare['low']['accuracy'],
        'header_low_accuracy': header['low']['accuracy'],
        'low_accuracy_change': header['low']['accuracy'] - bare['low']['accuracy'],
        'no_header_random_accuracy': bare['random_five_trial_mean']['accuracy'],
        'header_random_accuracy': header['random_five_trial_mean']['accuracy'],
        'random_accuracy_change': header['random_five_trial_mean']['accuracy'] - bare['random_five_trial_mean']['accuracy'],
        'header_strategy_difference': header['accuracy_difference'],
        'no_header_low_symbols_per_request': bare['low']['added_symbols_per_request'],
        'header_low_symbols_per_request': header['low']['added_symbols_per_request'],
        'no_header_random_symbols_per_request': bare['random_five_trial_mean']['added_symbols_per_request'],
        'header_random_symbols_per_request': header['random_five_trial_mean']['added_symbols_per_request'],
    })
result = {
    'scope': 'development-only 16-bit feedback header sensitivity; same images and cached data outcomes',
    'feedback_payload_bits': {'without_header': 24, 'with_header': 40},
    'feedback_symbols_per_attempt': {'without_header': 23, 'with_header': 31},
    'header_format': '4-bit type=1 plus 12-bit source image index; reference design',
    'feedback_seeds': [r['feedback_seed'] for r in rows],
    'low_accuracy_change_range': [min(r['low_accuracy_change'] for r in rows),
                                  max(r['low_accuracy_change'] for r in rows)],
    'random_accuracy_change_range': [min(r['random_accuracy_change'] for r in rows),
                                     max(r['random_accuracy_change'] for r in rows)],
    'rows': rows,
    'caution': 'Same 100 development images and cached physical data outcomes; no ACK, pilots, timeout duration or new holdout.'}
out = run / 'stage19_header16_sensitivity.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
