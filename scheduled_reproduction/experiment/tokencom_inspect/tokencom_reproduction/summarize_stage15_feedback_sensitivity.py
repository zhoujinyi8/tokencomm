"""Descriptive sensitivity across six feedback channel random seeds."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
paths = [root / 'runs/stage15_feedback_gate_summary.json'] + [
    root / f'runs/stage15_feedback_gate_seed{seed}.json'
    for seed in range(20261030, 20261035)]
cases = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
assert [x['feedback_seed'] for x in cases] == list(range(20261029, 20261035))
assert all(x['missing_tokens_across_seeds'] == 24320 for x in cases)
result = {
    'scope': 'development feedback seed sensitivity; same images and data retransmission outcomes',
    'feedback_seeds': [x['feedback_seed'] for x in cases],
    'difference_range': [min(x['difference'] for x in cases),
                         max(x['difference'] for x in cases)],
    'rows': [{'feedback_seed': x['feedback_seed'],
              'low_accuracy': x['low_accuracy'],
              'random_mean_accuracy': x['random_five_trial_mean_accuracy'],
              'difference': x['difference']} for x in cases],
    'caution': 'These are repeated feedback noise draws on the same images and cached data outcomes.',
}
out = root / 'runs/stage15_feedback_sensitivity.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
