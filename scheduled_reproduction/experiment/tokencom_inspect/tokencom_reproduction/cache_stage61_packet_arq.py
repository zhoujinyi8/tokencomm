"""Independent matched-padded packet ARQ traces, with bounded pilot and replay gates."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

from packet_arq_accounting import packet_noise,replay_trace
from randomized_packetization import permutation_arrays
from smoke_stage25_sionna_link import CRCDecoder,CRCEncoder,ConvEncoder,Demapper,Mapper,ViterbiDecoder,tf

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes',type=float,default=10)
parser.add_argument('--pilot',action='store_true')
parser.add_argument('--smoke-and-pilot',action='store_true')
args=parser.parse_args();assert 0<args.max_minutes<=30
plan_path=run/'stage61_packet_arq_plan.json';plan=read(plan_path.name)
random=read('stage59_randomized_packetization_plan.json')
assert sha(run/'stage59_randomized_packetization_plan.json')==plan['source_randomization_plan_sha256']
assert tf.__version__=='2.15.1' and importlib.metadata.version('sionna')=='0.19.2'
assert np.__version__=='1.26.4'
for name,digest in plan['source_token_inputs'].items():assert sha(root/name)==digest
contract=read('stage61_arq_contract.json')
assert contract['passed']==8 and contract['failed']==0 and contract['noise_seed_combinations_verified']==614400
assert contract['helper_sha256']==sha(root/'packet_arq_accounting.py')
assert contract['test_script_sha256']==sha(root/'test_packet_arq_accounting.py')
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:images=json.loads(saved['metadata'].item())['rows']
identity=dict(plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),helper_sha256=sha(root/'packet_arq_accounting.py'),
              permutation_helper_sha256=sha(root/'randomized_packetization.py'),
              component_source_sha256=sha(root/'smoke_stage25_sionna_link.py'),
              numpy=np.__version__,tensorflow=tf.__version__,sionna=importlib.metadata.version('sionna'))
lock_path=run/'stage61_packet_arq_lock_v2.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')


def frame(image,trial,si,noiseless=False):
    p,inverse=permutation_arrays(random['permutations'][image])
    packets=sent[image,p].reshape(4,64).T
    bits=((packets[...,None]>>np.arange(9,-1,-1))&1).astype(np.float32).reshape(64,40)
    crc_encoder=CRCEncoder('CRC11');crc_decoder=CRCDecoder(crc_encoder)
    encoder=ConvEncoder(gen_poly=['101','111']);decoder=ViterbiDecoder(encoder=encoder)
    coded=encoder(crc_encoder(tf.convert_to_tensor(bits)));assert coded.shape==(64,102)
    padded=tf.pad(coded,[[0,0],[0,2]])
    symbols=Mapper('qam',num_bits_per_symbol=4)(padded);assert symbols.shape==(64,26)
    demapper=Demapper(demapping_method='app',constellation_type='qam',num_bits_per_symbol=4)
    cap=plan['maximum_attempts_including_initial'];variance=np.float32(10**(-plan['snr_db'][si]/10))
    received=np.full((cap,64,4),-1,dtype=np.int16);valid=np.zeros((cap,64),bool);active_trace=np.zeros((cap,64),bool)
    crc_pending=np.ones(64,bool);oracle_pending=np.ones(64,bool)
    for attempt in range(1,cap+1):
        active=np.flatnonzero(crc_pending|oracle_pending)
        if not len(active):break
        noise=(np.zeros((len(active),26),np.complex64) if noiseless else
               packet_noise(plan['noise']['seed_base'],image,trial,si,active,attempt,float(variance)))
        channel=tf.gather(symbols,active)+tf.convert_to_tensor(noise)
        llr=demapper((channel,variance))
        decoded,crc_valid=crc_decoder(decoder(llr[...,:102]))
        raw=np.rint(decoded.numpy()).astype(np.int32).reshape(-1,4,10)
        rx=(raw*(1<<np.arange(9,-1,-1))).sum(axis=-1)
        good_crc=crc_valid.numpy().reshape(-1).astype(bool);good_payload=np.all(rx==packets[active],axis=1)
        received[attempt-1,active]=rx;valid[attempt-1,active]=good_crc;active_trace[attempt-1,active]=True
        crc_pending[active[good_crc]]=False;oracle_pending[active[good_payload]]=False
    policies=replay_trace(packets,received,valid,active_trace,cap)
    finals={}
    for policy,row in policies.items():
        chosen=np.asarray(row['packet_attempts'])-1
        packet_rx=received[chosen,np.arange(64)]
        finals[policy]=packet_rx.T.reshape(256)[inverse]
        assert row['final_token_errors']==int((finals[policy]!=sent[image]).sum())
    arrays=dict(transmitted_packets=packets,trace_received=received,trace_crc_valid=valid,trace_active=active_trace,
                **{f'final_{policy}':values for policy,values in finals.items()})
    return arrays,policies,int(active_trace.sum())


if args.smoke_and_pilot:
    checks=[]
    for image in (0,9):
        arrays,policies,observations=frame(image,0,4,True)
        assert observations==64
        assert all(v['total_packet_attempts']==64 and v['final_token_errors']==0 and v['censored_packets']==0 for v in policies.values())
        assert all(np.array_equal(arrays[f'final_{policy}'],sent[image]) for policy in policies)
        checks.append(dict(image_index=image,packet_observations=64,all_policies_first_attempt=True,token_errors=0))
    smoke=dict(identity,passed=2,failed=0,diagnostic_noiseless_frames=2,diagnostic_packet_observations=128,
               added_channel_noise=False,model_inference_performed=False,cases=checks)
    (run/'stage61_arq_noiseless_verification.json').write_text(json.dumps(smoke,indent=2),encoding='utf-8')
    print(json.dumps(smoke,indent=2),flush=True)
smoke=read('stage61_arq_noiseless_verification.json')
assert smoke['passed']==2 and all(smoke[k]==v for k,v in identity.items())
target=run/'stage61_packet_arq';target.mkdir(exist_ok=True)
coordinates=(plan['pilot_coordinates'] if args.pilot or args.smoke_and_pilot else
             [(i,t) for i in range(10) for t in range(5)])
deadline=time.monotonic()+60*args.max_minutes;budget=False
for image,trial in coordinates:
    for si,snr in enumerate(plan['snr_db']):
        out=target/f'snr{si}_image{image:02d}_trial{trial}.json'
        if out.exists():
            case=json.loads(out.read_text(encoding='utf-8'))
            assert case['lock_sha256']==sha(lock_path) and case['trace_arrays_sha256']==sha(out.with_suffix('.npz'))
            continue
        if time.monotonic()>deadline:budget=True;break
        arrays,policies,observations=frame(image,trial,si)
        npz=out.with_suffix('.npz');temporary=npz.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**arrays);os.replace(temporary,npz)
        case=dict(image_index=image,trial=trial,snr_index=si,snr_db=snr,source_sha256=images[image]['image_sha256'],
                  plan_sha256=sha(plan_path),lock_sha256=sha(lock_path),trace_arrays_sha256=sha(npz),
                  union_packet_observations=observations,policies=policies,model_inference_performed=False)
        temporary=out.with_suffix('.partial.json');temporary.write_text(json.dumps(case,indent=2),encoding='utf-8');os.replace(temporary,out)
        print(f'Padded ARQ snr={snr} image={image} trial={trial} union_packets={observations}',flush=True)
    if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json'))
progress=dict(identity,lock_sha256=sha(lock_path),completed_frame_experiments=len(files),expected_frame_experiments=300,
              complete=len(files)==300,pilot_expected=24,pilot_complete=all((target/f'snr{s}_image{i:02d}_trial{t}.json').exists()
                  for i,t in plan['pilot_coordinates'] for s in range(6)),budget_reached=budget,
              case_sha256={p.name:sha(p) for p in files})
(run/'stage61_packet_arq_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in progress.items() if k!='case_sha256'},indent=2))
