"""Cache the independent random-packet study without modifying the fixed baseline."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from randomized_packetization import permutation_arrays,restore_reception
from smoke_stage25_sionna_link import simulate

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes',type=float,default=12)
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--smoke-and-cache',action='store_true')
args=parser.parse_args();assert 0<args.max_minutes<=30
root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
spec_path=run/'stage60_execution_spec.json';spec=read(spec_path.name)
plan_path=run/'stage59_randomized_packetization_plan.json';plan=read(plan_path.name)
assert spec['preregistered_plan_sha256']==sha(plan_path)
assert spec['phy_cache_script_sha256']==sha(Path(__file__))
assert spec['packetization_helper_sha256']==sha(root/'randomized_packetization.py')
assert spec['phy_simulator_sha256']==sha(root/'smoke_stage25_sionna_link.py')
assert sha(run/'stage59_development_recovery_audit.json')==spec['baseline_complete_audit_sha256']
for name,digest in spec['input_sha256'].items():assert sha(root/name)==digest
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:images=json.loads(saved['metadata'].item())['rows']
identity={'script_sha256':sha(Path(__file__)),'plan_sha256':sha(spec_path),
          'preregistered_plan_sha256':sha(plan_path),'baseline_complete_audit_sha256':spec['baseline_complete_audit_sha256'],
          'helper_sha256':sha(root/'randomized_packetization.py'),'simulator_sha256':spec['phy_simulator_sha256']}
if args.smoke or args.smoke_and_cache:
    results=[]
    for image in (0,9):
        p,inverse=permutation_arrays(plan['permutations'][image])
        metadata,rx,oracle,crc=simulate(202611260+image,80.,tokens=sent[image,p],return_arrays=True)
        original,mask,crc_mask=restore_reception(rx,oracle,crc,inverse)
        assert np.array_equal(original,sent[image]) and not mask.any() and not crc_mask.any()
        assert metadata['bit_errors']==metadata['token_errors']==0
        results.append({'image_index':image,'snr_db':80,'bit_errors':0,'token_errors':0})
    result=dict(identity,passed=2,failed=0,model_inference_performed=False,diagnostic_phy_simulations=2,
                study_phy_simulations=0,method='high-SNR round trip; not a mathematically noiseless channel',cases=results)
    (run/'stage60_phy_roundtrip_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if args.smoke:raise SystemExit(0)
smoke=read('stage60_phy_roundtrip_smoke.json')
assert smoke['passed']==2 and all(smoke[k]==v for k,v in identity.items())
lock_path=run/'stage60_randomized_phy_lock.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
target=run/'stage60_randomized_phy';target.mkdir(exist_ok=True)
deadline=time.monotonic()+60*args.max_minutes;budget=False
for image in range(10):
    p,inverse=permutation_arrays(plan['permutations'][image])
    for trial in range(5):
        for si,snr in enumerate(spec['snr_db']):
            out=target/f'snr{si}_image{image:02d}_trial{trial}.npz'
            if out.exists():
                with np.load(out) as saved:assert saved['lock_sha256'].item()==sha(lock_path)
                continue
            if time.monotonic()>deadline:budget=True;break
            metadata,rx,oracle,crc=simulate(spec['phy_seed_base']+100*image+trial,float(snr),tokens=sent[image,p],return_arrays=True)
            canonical,oracle_mask,crc_mask=restore_reception(rx,oracle,crc,inverse)
            temporary=out.with_suffix('.partial.npz')
            np.savez_compressed(temporary,received=canonical,oracle_token_mask=oracle_mask,crc_token_mask=crc_mask,
                permuted_received=rx.reshape(256),oracle_bad_packets=oracle,crc_bad_packets=crc,
                source_sha256=images[image]['image_sha256'],plan_sha256=sha(spec_path),lock_sha256=sha(lock_path),
                permutation_sha256=plan['permutations'][image]['int64_little_endian_sha256'],metadata=json.dumps(metadata))
            os.replace(temporary,out)
            print(f'Random PHY snr={snr} image={image+1}/10 trial={trial+1}/5',flush=True)
        if budget:break
    if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].npz'))
result=dict(identity,lock_sha256=sha(lock_path),completed_cases=len(files),expected_cases=300,complete=len(files)==300,
            counts_by_snr={str(s):sum(p.name.startswith(f'snr{i}_') for p in files) for i,s in enumerate(spec['snr_db'])},
            budget_reached=budget,case_sha256={p.name:sha(p) for p in files})
(run/'stage60_randomized_phy_progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='case_sha256'},indent=2))
