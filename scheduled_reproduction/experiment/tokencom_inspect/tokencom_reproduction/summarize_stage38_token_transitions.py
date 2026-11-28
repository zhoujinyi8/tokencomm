"""Account exactly for repairs and collateral token errors."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
run = root/'runs'
link = np.load(run/'stage26_official_token_link_snr8_5trials.npz')
totals = {name: {k: 0 for k in ['masked_received_correct','masked_received_wrong','wrong_to_correct',
                               'correct_to_wrong','correct_to_correct_masked','wrong_to_wrong',
                               'direct_correct_tokens','correct_total_tokens']}
          for name in ('oracle_data_error','receiver_crc_failure')}
rows = []
for trial in range(5):
    for image in range(10):
        path = run/f'stage38_token_transitions_trial{trial}_image{image}.json'
        if not path.exists(): continue
        current = json.loads(path.read_text(encoding='utf-8'))
        old = json.loads((run/f'stage35_clip_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
        assert (current['trial'], current['image_index']) == (trial,image)
        for now, prior, mask_name in zip(current['rows'],old['rows'],('oracle_token_mask','crc_token_mask')):
            assert now['correct_total_tokens'] == prior['correct_total_tokens']
            assert abs(now['mse_to_clean_vqgan']-prior['mse_to_clean_vqgan']) < 1e-8
            final = np.array(now['final_token_ids'])
            before = link['received'][trial,image]
            tx = link['transmitted'][image]
            mask = link[mask_name][trial,image].astype(bool)
            assert np.array_equal(final[~mask],before[~mask])
            assert now['wrong_to_correct'] == int(((before!=tx)&(final==tx)).sum())
            assert now['correct_to_wrong'] == int(((before==tx)&(final!=tx)).sum())
            assert now['correct_total_tokens']-now['direct_correct_tokens'] == now['wrong_to_correct']-now['correct_to_wrong']
            for key in totals[now['mask_strategy']]: totals[now['mask_strategy']][key] += now[key]
        rows.append({'trial':trial,'image_index':image,'rows':current['rows']})
result={'pairs':len(rows),'totals':totals,'scope':'available saved development-image receptions; token transitions counted across saved positions',
        'completed_cases': [{'trial':r['trial'],'image_index':r['image_index']} for r in rows]}
(run/'stage38_token_transition_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
