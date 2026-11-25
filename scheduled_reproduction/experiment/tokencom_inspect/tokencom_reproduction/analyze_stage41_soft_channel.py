"""Describe coded-channel proxy scores within CRC-failed token positions."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

root=Path(__file__).resolve().parent
run=root/'runs'
channel=np.load(run/'stage41_soft_channel.npz')
old=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
llr=channel['coded_llr']
encoded=channel['reencoded_bits']
# Sionna demapper LLR is log(P(bit=1)/P(bit=0)). The two output bits
# associated with each information bit share convolutional state across tokens.
signed=(2*encoded-1)*llr
features={
    'negative_mean_abs_coded_llr':-np.abs(llr[...,:80]).reshape(5,10,64,4,20).mean(-1),
    'reencoded_path_observation_cost':np.logaddexp(0,-signed[...,:80]).reshape(5,10,64,4,20).mean(-1),
    'hard_demapper_path_disagreement':((llr[...,:80]>0)!=(encoded[...,:80]>0.5)).reshape(5,10,64,4,20).mean(-1)}
def auc(labels,score):
    n=int(labels.sum()); m=len(labels)-n
    if n==0 or m==0:return None
    return float((rankdata(score)[labels].sum()-n*(n+1)/2)/(n*m))
labels=old['received']!=old['transmitted'][None,:,:]
crc=old['crc_token_mask']
rows=[]
for name,score in features.items():
    score=score.transpose(0,1,3,2).reshape(5,10,256)
    per_image=[]
    for image in range(10):
        mask=crc[:,image]
        per_image.append({'image_index':image,'auc_over_five_trials':auc(labels[:,image][mask],score[:,image][mask])})
    selected=crc
    actual=labels[selected]
    values=score[selected]
    # Fixed descriptive fractions; no operating threshold is selected.
    top=[]
    for fraction in (.05,.1,.25):
        k=max(1,int(np.ceil(len(values)*fraction)))
        ids=np.argsort(-values,kind='stable')[:k]
        top.append({'fraction':fraction,'selected':k,'errors_selected':int(actual[ids].sum()),
                    'precision':float(actual[ids].mean()),'error_recall':float(actual[ids].sum()/actual.sum())})
    rows.append({'feature':name,'pooled_auc_descriptive':auc(actual,values),
                 'mean_auc_over_images':float(np.mean([r['auc_over_five_trials'] for r in per_image if r['auc_over_five_trials'] is not None])),
                 'top_risk_fractions':top,'per_image':per_image})
result={'scope':'10 development images x 5 fixed PHY receptions; diagnostics restricted to CRC-failed token positions',
        'crc_masked_positions':int(crc.sum()),'wrong_positions':int(labels[crc].sum()),
        'baseline_error_fraction':float(labels[crc].mean()),'score_direction':'larger is more risky',
        'note':'not decoded-token posteriors; convolutional memory couples neighboring token segments; pooled positions are not independent samples',
        'features':rows}
(run/'stage41_soft_channel_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({**result,'features':[{k:v for k,v in r.items() if k!='per_image'} for r in rows]},indent=2))
