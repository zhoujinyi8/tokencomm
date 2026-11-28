"""Check the stage19 feedback Es/N0 sensitivity table from saved results."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
rows = []
for snr in (6, 8, 10):
    for header in (0, 16):
        name = (f'stage19_snr{snr}_header{header}.json' if snr != 8 else
                'stage16_feedback_retry_summary.json' if header == 0 else
                'stage19_header16_seed20261029.json')
        case = json.loads((run / name).read_text(encoding='utf-8'))
        if 'feedback_snr_db' in case:
            assert case['feedback_snr_db'] == snr
        assert case.get('header_bits', 0) == header
        assert case['feedback_seed'] == 20261029
        rows.append({'feedback_snr_db': snr, 'header_bits': header,
                     'low_accuracy': case['low']['accuracy'],
                     'random_accuracy': case['random_five_trial_mean']['accuracy'],
                     'low_added_symbols_per_image': case['low']['added_symbols_per_request'],
                     'random_added_symbols_per_image': case['random_five_trial_mean']['added_symbols_per_request']})
out = run / 'stage19_header_snr_sensitivity.json'
out.write_text(json.dumps({'scope': 'development-only feedback Es/N0 sweep; data channel stays Rayleigh 8 dB',
                           'feedback_seed': 20261029, 'rows': rows}, indent=2), encoding='utf-8')
print(out)
