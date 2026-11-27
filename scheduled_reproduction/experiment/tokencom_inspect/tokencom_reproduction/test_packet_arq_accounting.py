"""Hand-specified CRC false acceptance, parity failures, caps and noise identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

from packet_arq_accounting import packet_noise,packet_seed,replay_trace


def verify():
    tx=np.zeros((2,4),dtype=int);rx=np.zeros((3,2,4),dtype=int)
    rx[0,0,0]=1;valid=np.ones((3,2),bool);valid[0,1]=False
    active=np.array([[1,1],[1,1],[0,0]],dtype=bool)
    result=replay_trace(tx,rx,valid,active,3)
    crc=result['receiver_crc_arq_padded'];oracle=result['sender_oracle_arq_padded']
    assert crc['packet_attempts']==[1,2] and crc['accepted_crc_payload_errors']==1
    assert oracle['packet_attempts']==[2,1] and oracle['final_payload_error_packets']==0
    assert result['single_pass_padded']['total_packet_attempts']==2
    assert crc['forward_qam_symbols']==78 and crc['forward_wire_bits']==312 and crc['logical_feedback_events']==3
    bad=np.ones((3,1,4),dtype=int);flags=np.zeros((3,1),bool)
    cap=replay_trace(np.zeros((1,4),int),bad,flags,np.ones((3,1),bool),3)
    assert all(cap[p]['censored_packets']==1 and cap[p]['total_packet_attempts']==3
               for p in ('receiver_crc_arq_padded','sender_oracle_arq_padded'))
    try:replay_trace(np.zeros((1,4),int),bad,flags,np.array([[1],[1],[0]],bool),3)
    except AssertionError:pass
    else:raise AssertionError('Premature unresolved truncation accepted')
    try:replay_trace(np.zeros((1,4),int),bad,flags,np.array([[1],[0],[1]],bool),3)
    except AssertionError:pass
    else:raise AssertionError('Gap in attempt trace accepted')
    first=packet_noise(2026112700,0,0,0,[1,3],2,.1)
    reverse=packet_noise(2026112700,0,0,0,[3,1],2,.1)
    assert np.array_equal(first,reverse[::-1])
    assert not np.array_equal(first,packet_noise(2026112700,0,0,0,[1,3],3,.1))
    assert packet_seed(2026112700,0,0,0,0,1)==2026112701
    assert packet_seed(2026112700,9,4,5,63,32)==2026112700+9000000+400000+50000+6300+32
    offsets=(np.arange(10,dtype=np.int64)[:,None,None,None,None]*1000000+
             np.arange(5,dtype=np.int64)[None,:,None,None,None]*100000+
             np.arange(6,dtype=np.int64)[None,None,:,None,None]*10000+
             np.arange(64,dtype=np.int64)[None,None,None,:,None]*100+
             np.arange(1,33,dtype=np.int64)[None,None,None,None,:])
    assert offsets.size==614400 and (np.diff(offsets.ravel())>0).all()
    root=Path(__file__).parent
    return dict(passed=8,failed=0,noise_seed_combinations_verified=614400,model_inference_performed=False,physical_simulations_performed=0,
                helper_sha256=hashlib.sha256((root/'packet_arq_accounting.py').read_bytes()).hexdigest(),
                test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checks=['CRC false acceptance','parity-only failure','initial and wire ledger','cap retained',
                        'premature truncation refused','gapped trace refused','noise order invariance','seed identity'])


if __name__=='__main__':
    result=verify()
    (Path(__file__).parent/'runs/stage61_arq_contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
