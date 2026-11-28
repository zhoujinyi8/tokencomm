"""Reproduce saved PHY outputs before caching receiver-side coded-bit diagnostics."""
import json
from pathlib import Path
import numpy as np
from smoke_stage25_sionna_link import simulate

root=Path(__file__).resolve().parent
run=root/'runs'
old=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
llrs=[]
reencoded=[]
decoded=[]
for trial in range(5):
    trial_llr=[]
    trial_reencoded=[]
    trial_decoded=[]
    for image in range(10):
        result,rx,oracle,crc,soft=simulate(20261112+100*image+trial,8.,tokens=old['transmitted'][image],return_soft=True)
        assert np.array_equal(rx.reshape(-1),old['received'][trial,image])
        assert np.array_equal(np.tile(oracle,4),old['oracle_token_mask'][trial,image])
        assert np.array_equal(np.tile(crc,4),old['crc_token_mask'][trial,image])
        trial_llr.append(soft['coded_llr'])
        trial_reencoded.append(soft['reencoded_decoded_bits'])
        trial_decoded.append(soft['decoded_crc_bits'])
    llrs.append(trial_llr)
    reencoded.append(trial_reencoded)
    decoded.append(trial_decoded)
    print(f'finished trial={trial}',flush=True)
metadata={'scope':'same 50 saved development receptions; exact received tokens and masks checked',
          'coded_llr_shape':'5 trials x 10 images x 64 packets x 102 coded bits',
          'note':'coded-bit APP LLR is not a decoded information-bit posterior; Viterbi remains hard-output'}
np.savez_compressed(run/'stage41_soft_channel.npz',coded_llr=np.array(llrs),
                    reencoded_bits=np.array(reencoded),decoded_crc_bits=np.array(decoded),metadata=json.dumps(metadata))
