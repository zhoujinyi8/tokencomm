"""Check canonical ID and packet-mask restoration with predetermined outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from randomized_packetization import permutation_arrays,restore_reception


def verify():
    root=Path(__file__).parent
    plan=json.loads((root/'runs/stage59_randomized_packetization_plan.json').read_text(encoding='utf-8'))
    sent=np.arange(256,dtype=np.int64);checks=0
    for row in plan['permutations']:
        p,inverse=permutation_arrays(row)
        restored,mask,crc=restore_reception(sent[p],np.zeros(64,bool),np.zeros(64,bool),inverse)
        assert np.array_equal(restored,sent) and not mask.any() and not crc.any();checks+=1
        corrupted=sent[p].copy();corrupted[3]=999
        oracle=np.zeros(64,bool);oracle[3]=True
        crc_packets=np.zeros(64,bool);crc_packets[7]=True
        restored,mask,crc=restore_reception(corrupted,oracle,crc_packets,inverse)
        assert np.flatnonzero(restored!=sent).tolist()==[int(p[3])]
        assert set(np.flatnonzero(mask))==set(p[[3,67,131,195]])
        assert set(np.flatnonzero(crc))==set(p[[7,71,135,199]])
        assert mask[p[3]] and not np.any(mask&crc);checks+=1
    bad=dict(plan['permutations'][0]);bad['inverse_permutation']=list(range(256))
    try:permutation_arrays(bad)
    except AssertionError:checks+=1
    else:raise AssertionError('Wrong inverse accepted')
    bad=dict(plan['permutations'][0]);bad['seed']+=1
    try:permutation_arrays(bad)
    except AssertionError:checks+=1
    else:raise AssertionError('Wrong RNG identity accepted')
    return dict(passed=checks,failed=0,model_inference_performed=False,physical_simulations_performed=0,
                helper_sha256=hashlib.sha256((root/'randomized_packetization.py').read_bytes()).hexdigest(),
                test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    result=verify()
    (Path(__file__).parent/'runs/stage60_packetization_contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
