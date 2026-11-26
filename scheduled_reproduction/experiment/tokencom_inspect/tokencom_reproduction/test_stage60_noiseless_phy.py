"""Exact noiseless CRC/conv/QAM round trip with positive demapper variance."""
import hashlib
import json
from pathlib import Path

import numpy as np

from randomized_packetization import permutation_arrays,restore_reception
from smoke_stage25_sionna_link import (CRCDecoder,CRCEncoder,ConvEncoder,Demapper,
                                      Mapper,ViterbiDecoder,tf)

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((run/'stage60_execution_spec.json').read_text(encoding='utf-8'))
plan=json.loads((run/'stage59_randomized_packetization_plan.json').read_text(encoding='utf-8'))
assert spec['preregistered_plan_sha256']==sha(run/'stage59_randomized_packetization_plan.json')
assert sha(root/'smoke_stage25_sionna_link.py')==spec['phy_simulator_sha256']
assert sha(root/'randomized_packetization.py')==spec['packetization_helper_sha256']
source=run/'stage26_official_token_link_snr8_5trials.npz'
assert sha(source)==spec['input_sha256'][source.relative_to(root).as_posix()]
with np.load(source) as saved:sent=saved['transmitted'].copy()
results=[]
for image in (0,9):
    p,inverse=permutation_arrays(plan['permutations'][image])
    packets=sent[image,p].reshape(1,4,64).transpose(0,2,1)
    bits=((packets[...,None]>>np.arange(9,-1,-1))&1).astype(np.float32).reshape(1,64,40)
    crc_encoder=CRCEncoder('CRC11');crc_decoder=CRCDecoder(crc_encoder)
    encoder=ConvEncoder(gen_poly=['101','111']);decoder=ViterbiDecoder(encoder=encoder)
    coded=encoder(crc_encoder(tf.convert_to_tensor(bits)))
    symbols=Mapper('qam',num_bits_per_symbol=4)(tf.reshape(coded,[1,-1,4]))
    # No channel noise is added. The positive variance defines finite soft LLRs.
    llr=Demapper(demapping_method='app',constellation_type='qam',num_bits_per_symbol=4)((symbols,0.01))
    decoded,valid=crc_decoder(decoder(tf.reshape(llr,[1,64,102])))
    restored_bits=np.rint(decoded.numpy()).astype(np.int32).reshape(1,64,4,10)
    received=(restored_bits*(1<<np.arange(9,-1,-1))).sum(axis=-1).transpose(0,2,1).reshape(256)
    crc_bad=~valid.numpy().reshape(64).astype(bool)
    oracle=(packets.reshape(64,4)!=received.reshape(4,64).T).any(axis=1)
    canonical,mask,crc_mask=restore_reception(received,oracle,crc_bad,inverse)
    assert np.array_equal(canonical,sent[image]) and not mask.any() and not crc_mask.any()
    assert not np.count_nonzero(bits-decoded.numpy())
    assert coded.shape[-1]==102 and int(np.prod(symbols.shape[1:]))==1632
    results.append(dict(image_index=image,payload_bit_errors=0,token_errors=0,crc_failures=0))
result=dict(passed=2,failed=0,script_sha256=sha(Path(__file__)),execution_spec_sha256=sha(run/'stage60_execution_spec.json'),
            simulator_sha256=spec['phy_simulator_sha256'],helper_sha256=spec['packetization_helper_sha256'],
            added_channel_noise=False,demapper_variance_assumption=0.01,model_inference_performed=False,
            diagnostic_noiseless_transmissions=2,formal_study_receptions=0,cases=results)
(run/'stage60_noiseless_phy_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
