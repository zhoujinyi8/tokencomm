"""Development audit: gate saved CRC generation using one masked-context forward pass."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS

root=Path(__file__).resolve().parent
official=root.parent.parent/'tokencom_official_reference'
sys.path.insert(0,str(official))
from Network.transformer import MaskTransformer

torch.set_num_threads(4)
run=root/'runs'
link=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
meta=json.loads(link['metadata'].item())
weight=root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth'
model=MaskTransformer(img_size=256,hidden_dim=768,codebook_size=1024,
                      depth=24,heads=16,mlp_dim=3072,dropout=.1).eval()
state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)['model_state_dict']
model.load_state_dict(state,strict=True,assign=True)
del state
# Prespecified diagnostic thresholds. No threshold is selected using these outcomes.
thresholds=[0.,0.5,0.9]
rows=[]
with torch.inference_mode():
    for trial in range(5):
        for image in range(10):
            rx=torch.from_numpy(link['received'][trial,image].copy()).long().reshape(1,16,16)
            mask=torch.from_numpy(link['crc_token_mask'][trial,image].copy()).bool().reshape(1,16,16)
            previous=json.loads((run/f'stage38_token_transitions_trial{trial}_image{image}.json').read_text(encoding='utf-8'))
            assert previous['rows'][1]['mask_strategy']=='receiver_crc_failure'
            candidate=torch.tensor(previous['rows'][1]['final_token_ids']).reshape_as(rx)
            cls=meta['rows'][trial*10+image]['class_name']
            context=rx.clone()
            context[mask]=1024
            logits=model(context,torch.tensor([IMAGENET_CLASS_IDS[cls]]),drop_label=torch.tensor([False]))
            probabilities=torch.softmax(logits*1.3,dim=-1)
            p_rx=probabilities.gather(-1,rx.reshape(1,256,1)).reshape_as(rx)
            p_candidate=probabilities.gather(-1,candidate.reshape(1,256,1)).reshape_as(rx)
            # Decisions above use only received tokens, CRC, class and saved CRC generation.
            # Sender truth is accessed below solely to score the fixed decisions.
            tx=torch.from_numpy(link['transmitted'][image].copy()).long().reshape_as(rx)
            direct_correct=int((rx==tx).sum())
            original_correct=int((candidate==tx).sum())
            policies=[]
            for threshold in thresholds:
                replace=mask&(candidate!=rx)&(p_candidate>p_rx)&(p_candidate>=threshold)
                final=rx.clone()
                final[replace]=candidate[replace]
                repaired=int(((rx!=tx)&(final==tx)).sum())
                harmed=int(((rx==tx)&(final!=tx)).sum())
                policies.append({'threshold':threshold,'replaced':int(replace.sum()),
                                 'repaired':repaired,'harmed':harmed,
                                 'correct_tokens':int((final==tx).sum()),
                                 'final_token_ids':final.reshape(-1).tolist()})
            rows.append({'trial':trial,'image_index':image,'class_name':cls,
                         'direct_correct':direct_correct,'original_crc_correct':original_correct,
                         'policies':policies,
                         'masked_p_received':p_rx[mask].tolist(),
                         'masked_p_candidate':p_candidate[mask].tolist()})
            print(f'DONE trial={trial} image={image}',flush=True)
result={'scope':'10 development images, 5 saved AWGN 8 dB receptions; class known; receiver CRC gate diagnostic',
        'thresholds':thresholds,'temperature_multiplier':1.3,
        'context':'all CRC-failed positions masked simultaneously; same context for candidate and received likelihood',
        'model_sha256':sha256(weight),'rows':rows}
(run/'stage40_confidence_gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for threshold in thresholds:
    selected=[next(p for p in r['policies'] if p['threshold']==threshold) for r in rows]
    print(json.dumps({'threshold':threshold,**{k:sum(p[k] for p in selected) for k in ('replaced','repaired','harmed','correct_tokens')}}))
