"""Verify saved frozen recovery semantics without computing accuracy metrics."""
import json
from pathlib import Path
import numpy as np
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs'
lock_path=run/'stage46_recovery_lock.json'
lock=json.loads(lock_path.read_text(encoding='utf-8'))
assert sha256(root/'run_stage46_frozen_recovery.py')==lock['runner_sha256']
assert sha256(run/'stage44_frozen_protocol.json')==lock['protocol_sha256']
checked=[]
for path in sorted((run/'stage46_holdout_recovery').glob('image[0-9][0-9][0-9]_trial[0-4].npz')):
    image=int(path.stem[5:8]);trial=int(path.stem[-1])
    phy=run/f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz'
    with np.load(phy) as x,np.load(path) as saved:
        rx=x['received'];crc=x['crc_token_mask'].astype(bool);oracle=x['oracle_token_mask'].astype(bool)
        assert np.array_equal(saved['direct'],rx)
        assert saved['source_sha256'].item()==x['source_sha256'].item()
        assert saved['source_phy_sha256'].item()==sha256(phy)
        assert saved['recovery_lock_sha256'].item()==sha256(lock_path)
        for name,mask in [('oracle',oracle),('crc',crc)]:
            assert saved[name].shape==(256,) and saved[name].min()>=0 and saved[name].max()<1024
            assert np.array_equal(saved[name][~mask],rx[~mask])
        signed=(2*x['reencoded_bits']-1)*x['coded_llr']
        cost=np.logaddexp(0,-signed[:,:80]).reshape(64,4,20).mean(-1).T.reshape(256)
        ids=np.flatnonzero(crc);ordered=ids[np.argsort(-cost[ids],kind='stable')]
        for fraction in (.05,.1,.25):
            selected=ordered[:max(1,int(np.ceil(len(ids)*fraction)))] if len(ids) else ordered
            expected=rx.copy();expected[selected]=saved['crc'][selected]
            assert np.array_equal(saved[f'soft_gate_{fraction}'],expected)
        assert all(0<=v<=32 for v in json.loads(saved['steps'].item()).values())
    checked.append({'image_index':image,'trial':trial})
result={'checked_cases':len(checked),'all_checks_passed':True,'performance_metrics_computed':False,
        'checks':['execution lock and input hash','direct reception identity','unmasked positions preserved',
                  'codebook range','frozen per-reception gate selection','step budget'], 'cases':checked}
(run/'stage46_output_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
