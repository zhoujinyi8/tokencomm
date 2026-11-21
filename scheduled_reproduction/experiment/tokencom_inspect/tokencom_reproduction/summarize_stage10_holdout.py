"""Descriptive roll-up of a resumable frozen holdout evaluation.

Repeated channel seeds reuse the same images; token totals are not independent
sample counts and this script does not compute significance tests.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    progress = json.loads((args.run / 'progress.json').read_text(encoding='utf-8'))
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if not status['test_accessed'] or status['cases_completed'] != len(progress['cases']):
        raise ValueError('Holdout status/progress mismatch')
    groups = defaultdict(list)
    for case in progress['cases']:
        if case['images'] != len(case['rows']): raise ValueError('Image count mismatch')
        if sum(row['missing_tokens'] for row in case['rows']) != case['missing_tokens']:
            raise ValueError('Missing token count mismatch')
        for method, count in case['correct_counts'].items():
            if sum(row['correct'][method] for row in case['rows']) != count:
                raise ValueError(f'Correct token count mismatch: {method}')
        groups[case['per']].append(case)
    summaries = []
    for per, cases in sorted(groups.items()):
        missing = sum(case['missing_tokens'] for case in cases)
        methods = cases[0]['correct_counts']
        correct = {method: sum(case['correct_counts'][method] for case in cases)
                   for method in methods}
        summaries.append({
            'per': per, 'seeds_completed': [case['seed'] for case in cases],
            'images_reused_per_seed': cases[0]['images'], 'missing_tokens_across_seeds': missing,
            'correct_counts': correct,
            'accuracy': {method: count / missing for method, count in correct.items()},
        })
    result = {'scope': 'frozen stage10 holdout', 'complete': status['complete'],
              'cases_completed': status['cases_completed'],
              'cases_expected': status['cases_expected'],
              'repeated_image_caution': 'The same images are reused across channel seeds.',
              'summaries': summaries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
