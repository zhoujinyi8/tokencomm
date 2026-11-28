"""Check repair/damage accounting against hand-constructed token outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_multisnr_token_transitions import count_transitions


def verify():
    sent=np.array([0,1,2,3]); mask=np.array([True,True,True,False])
    clean=count_transitions(sent,sent,sent,np.zeros(4,dtype=bool))
    assert clean['repaired']==clean['damaged']==clean['net_correct_gain']==0
    rx=np.array([9,1,2,3])
    repaired=count_transitions(sent,rx,sent,mask)
    assert repaired['repaired']==1 and repaired['damaged']==0 and repaired['net_correct_gain']==1
    balanced=count_transitions(sent,rx,np.array([0,9,2,3]),mask)
    assert balanced['repaired']==1 and balanced['damaged']==1 and balanced['net_correct_gain']==0
    degraded=count_transitions(sent,rx,np.array([8,9,2,3]),mask)
    assert degraded['repaired']==0 and degraded['damaged']==1 and degraded['wrong_to_different_wrong']==1
    assert degraded['wrong_before']==1 and degraded['wrong_after']==2 and degraded['net_correct_gain']==-1
    try:
        count_transitions(sent,rx,np.array([0,1,2,9]),mask)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unmasked modification accepted')
    root=Path(__file__).parent
    return dict(passed=5,failed=0,network_used=False,model_inference_performed=False,
                script_sha256=hashlib.sha256((root/'summarize_multisnr_token_transitions.py').read_bytes()).hexdigest(),
                test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checks=['identity','one repair','repair cancelled by damage','different wrong token and damage','unmasked modification rejected'])


if __name__ == '__main__':
    result=verify()
    (Path(__file__).parent/'runs/stage59_token_transition_contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
