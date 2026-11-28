"""Actual Sionna variable-packet frames with all stopping traces preserved."""
import argparse,hashlib,importlib.metadata,json,os,time
from pathlib import Path
import numpy as np
from cheng_byte_transport import byte_packets,packet_bytes,byte_packet_noise
from packet_arq_accounting import replay_trace
from smoke_stage25_sionna_link import CRCDecoder,CRCEncoder,ConvEncoder,Demapper,Mapper,ViterbiDecoder,tf
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=15);p.add_argument('--pilot',action='store_true');args=p.parse_args();assert 0<args.max_minutes<=25
plan_path=run/'stage66_transport_plan.json';plan=read(plan_path.name)
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
assert tf.__version__=='2.15.1' and importlib.metadata.version('sionna')=='0.19.2' and np.__version__=='1.26.4'
assert sha(run/'stage66_transport_contract.json')==plan['contract_sha256']
assert sha(run/'stage66_rate_grid_audit.json')==plan['grid_audit_sha256']
for n,h in plan['prior_frozen_artifacts'].items():assert sha(run/n)==h
tf.config.threading.set_intra_op_parallelism_threads(plan['tensorflow_threads']);tf.config.threading.set_inter_op_parallelism_threads(1)
crc_encoder=CRCEncoder('CRC11');crc_decoder=CRCDecoder(crc_encoder);encoder=ConvEncoder(gen_poly=['101','111']);decoder=ViterbiDecoder(encoder=encoder)
mapper=Mapper('qam',num_bits_per_symbol=4);demapper=Demapper(demapping_method='app',constellation_type='qam',num_bits_per_symbol=4)
streams={v['image_index']:(r/v['container_path']).read_bytes() for v in plan['streams']}
for v in plan['streams']:assert sha(r/v['container_path'])==v['container_sha256']
def frame(image,trial,si,noiseless=False):
    original=streams[image];sent=byte_packets(original);count=len(sent);cap=plan['maximum_attempts']
    bits=((sent[...,None]>>np.arange(9,-1,-1))&1).astype(np.float32).reshape(count,40)
    coded=encoder(crc_encoder(tf.convert_to_tensor(bits)));assert coded.shape==(count,102)
    symbols=mapper(tf.pad(coded,[[0,0],[0,2]]));assert symbols.shape==(count,26)
    received=np.full((cap,count,4),-1,np.int16);valid=np.zeros((cap,count),bool);active_trace=np.zeros((cap,count),bool)
    crc_pending=np.ones(count,bool);oracle_pending=np.ones(count,bool);variance=np.float32(10**(-plan['snr_db'][si]/10))
    for attempt in range(1,cap+1):
        active=np.flatnonzero(crc_pending|oracle_pending)
        if not len(active):break
        noise=np.zeros((len(active),26),np.complex64) if noiseless else byte_packet_noise(plan['noise_seed_base'],image,trial,si,active,attempt,float(variance))
        channel=tf.gather(symbols,active)+tf.convert_to_tensor(noise);llr=demapper((channel,variance));decoded,crc_valid=crc_decoder(decoder(llr[...,:102]))
        raw=np.rint(decoded.numpy()).astype(np.int32).reshape(-1,4,10);rx=(raw*(1<<np.arange(9,-1,-1))).sum(axis=-1)
        good_crc=crc_valid.numpy().reshape(-1).astype(bool);good_payload=np.all(rx==sent[active],axis=1)
        received[attempt-1,active]=rx;valid[attempt-1,active]=good_crc;active_trace[attempt-1,active]=True
        crc_pending[active[good_crc]]=False;oracle_pending[active[good_payload]]=False
    policies=replay_trace(sent,received,valid,active_trace,cap);finals={}
    for name,v in policies.items():
        chosen=np.asarray(v['packet_attempts'])-1;final=received[chosen,np.arange(count)];container=packet_bytes(final,len(original))
        v['container_byte_errors']=sum(a!=b for a,b in zip(container,original));v['container_exact']=container==original
        v['complete_and_exact']=v['container_exact'] and v['censored_packets']==0
        v['semantic_token_errors_not_measured']=True;finals['final_'+name]=final
    arrays=dict(transmitted_packets=sent,trace_received=received,trace_crc_valid=valid,trace_active=active_trace,**finals)
    return arrays,policies,int(active_trace.sum())
smoke_path=run/'stage66_transport_noiseless.json'
if not smoke_path.exists():
    cases=[]
    for image in plan['diagnostic_noiseless_images']:
        arrays,policies,observations=frame(image,0,4,True)
        assert all(v['container_exact'] and v['censored_packets']==0 and v['total_packet_attempts']==len(arrays['transmitted_packets']) for v in policies.values())
        cases.append(dict(image_index=image,packets=len(arrays['transmitted_packets']),observations=observations,all_policies_exact=True))
    smoke=dict(plan_sha256=sha(plan_path),worker_sha256=sha(__file__),passed=2,failed=0,cases=cases,added_channel_noise=False)
    smoke_path.write_text(json.dumps(smoke,indent=2),encoding='utf-8')
smoke=read(smoke_path.name);assert smoke['plan_sha256']==sha(plan_path) and smoke['worker_sha256']==sha(__file__) and smoke['passed']==2
target=run/'stage66_transport_cases';target.mkdir(exist_ok=True);deadline=time.monotonic()+args.max_minutes*60;budget=False
coordinates=plan['pilot_coordinates'] if args.pilot else [(i,t) for i in range(10) for t in range(5)]
for image,trial in coordinates:
    for si,snr in enumerate(plan['snr_db']):
        path=target/f'snr{si}_image{image:02d}_trial{trial}.json'
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'));assert previous['plan_sha256']==sha(plan_path) and previous['worker_sha256']==sha(__file__) and sha(path.with_suffix('.npz'))==previous['trace_arrays_sha256'];continue
        if time.monotonic()>=deadline:budget=True;break
        arrays,policies,observations=frame(image,trial,si)
        array_path=path.with_suffix('.npz');temp=array_path.with_suffix('.partial.npz');np.savez_compressed(temp,**arrays);os.replace(temp,array_path)
        value=dict(image_index=image,trial=trial,snr_index=si,snr_db=snr,plan_sha256=sha(plan_path),worker_sha256=sha(__file__),trace_arrays_sha256=sha(array_path),
            source_container_sha256=next(v['container_sha256'] for v in plan['streams'] if v['image_index']==image),
            packet_count=len(arrays['transmitted_packets']),union_packet_observations=observations,policies=policies,entropy_decoder_called=False)
        temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)
        print(f'Cheng byte transport image={image} trial={trial} snr={snr} observations={observations}',flush=True)
    if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json'))
progress=dict(plan_sha256=sha(plan_path),worker_sha256=sha(__file__),completed_frames=len(files),expected_frames=300,complete=len(files)==300,
              pilot_complete=all((target/f'snr{s}_image{i:02d}_trial{t}.json').exists() for i,t in plan['pilot_coordinates'] for s in range(6)),
              budget_reached=budget,case_sha256={f.name:sha(f) for f in files},entropy_decoder_called=False,paper_baseline_ready=False)
(run/'stage66_transport_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in progress.items() if k!='case_sha256'}))
