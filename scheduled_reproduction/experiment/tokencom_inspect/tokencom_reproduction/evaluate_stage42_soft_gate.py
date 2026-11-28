"""Development-only per-reception soft-channel ranking of CRC candidates."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
run=root/'runs'
channel=np.load(run/'stage41_soft_channel.npz')
link=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
llr=channel['coded_llr']; bits=channel['reencoded_bits']
cost=np.logaddexp(0,-(2*bits-1)*llr)[...,:80].reshape(5,10,64,4,20).mean(-1)
cost=cost.transpose(0,1,3,2).reshape(5,10,256)
rows=[]
for trial in range(5):
    for image in range(10):
        rx=link['received'][trial,image]
        mask=link['crc_token_mask'][trial,image].astype(bool)
        candidate=np.array(json.loads((run/f'stage38_token_transitions_trial{trial}_image{image}.json').read_text(encoding='utf-8'))['rows'][1]['final_token_ids'])
        ids=np.flatnonzero(mask)
        ordered=ids[np.argsort(-cost[trial,image,ids],kind='stable')]
        # All fixed fractions reported. Ranking occurs inside each reception,
        # never using transmitted values or other images' scores.
        for fraction in (.05,.1,.25):
            selected=ordered[:max(1,int(np.ceil(len(ids)*fraction)))] if len(ids) else ordered
            final=rx.copy(); final[selected]=candidate[selected]
            tx=link['transmitted'][image]
            repaired=int(((rx!=tx)&(final==tx)).sum())
            harmed=int(((rx==tx)&(final!=tx)).sum())
            rows.append({'trial':trial,'image_index':image,'fraction':fraction,
                         'selected':len(selected),'changed':int((rx!=final).sum()),
                         'repaired':repaired,'harmed':harmed,'correct':int((final==tx).sum()),
                         'final_token_ids':final.tolist()})
totals=[]
for fraction in (.05,.1,.25):
    selected=[r for r in rows if r['fraction']==fraction]
    totals.append({'fraction':fraction,**{k:sum(r[k] for r in selected) for k in ('selected','changed','repaired','harmed','correct')}})
result={'scope':'development-only CRC gate, per-reception ranked path cost; saved original CRC generation candidates',
        'fractions':[.05,.1,.25],'chosen_fraction':None,'totals':totals,'rows':rows}
(run/'stage42_soft_gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(totals,indent=2))
