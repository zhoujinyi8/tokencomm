"""Aggregate saved transitions and split CRC harm by data-error packet status."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
run=root/'runs'
link=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
cases=[]
for trial in range(5):
    for image in range(10):
        path=run/f'stage38_token_transitions_trial{trial}_image{image}.json'
        if not path.exists():continue
        data=json.loads(path.read_text(encoding='utf-8'))
        tx=link['transmitted'][image]
        rx=link['received'][trial,image]
        oracle_mask=link['oracle_token_mask'][trial,image].astype(bool)
        crc_mask=link['crc_token_mask'][trial,image].astype(bool)
        prior=json.loads((run/f'stage35_clip_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
        rows=[]
        for row,old,mask in zip(data['rows'],prior['rows'],(oracle_mask,crc_mask)):
            final=np.array(row['final_token_ids'])
            repaired=(rx!=tx)&(final==tx)
            harmed=(rx==tx)&(final!=tx)
            assert np.array_equal(final[~mask],rx[~mask])
            assert int(repaired.sum())==row['wrong_to_correct']
            assert int(harmed.sum())==row['correct_to_wrong']
            assert row['correct_total_tokens']==old['correct_total_tokens']
            assert abs(row['mse_to_clean_vqgan']-old['mse_to_clean_vqgan'])<1e-8
            rows.append({'strategy':row['mask_strategy'],'repaired':int(repaired.sum()),
                         'harmed':int(harmed.sum()),
                         'harmed_in_data_error_packets':int((harmed&oracle_mask).sum()),
                         'harmed_outside_data_error_packets':int((harmed&~oracle_mask).sum()),
                         'net_correct_change':int(repaired.sum()-harmed.sum())})
        cases.append({'trial':trial,'image_index':image,'class_name':data['class_name'],'rows':rows})
keys=['repaired','harmed','harmed_in_data_error_packets','harmed_outside_data_error_packets','net_correct_change']
totals={s:{k:sum(r[k] for c in cases for r in c['rows'] if r['strategy']==s) for k in keys}
        for s in ('oracle_data_error','receiver_crc_failure')}
complete=[i for i in range(10) if sum(c['image_index']==i for c in cases)==5]
result={'pairs':len(cases),'complete_five_trial_images':complete,'totals_over_saved_cases':totals,'cases':cases}
(run/'stage39_transition_attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
