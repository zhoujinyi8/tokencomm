"""Independent exact trace replay; partial SNR groups never receive delivery means."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from cheng_byte_transport import byte_packets,packet_bytes
from packet_arq_accounting import POLICIES,replay_trace
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_transport_plan.json');progress=read('stage66_transport_progress.json');target=run/'stage66_transport_cases'
assert progress['plan_sha256']==sha(run/'stage66_transport_plan.json') and progress['worker_sha256']==sha(r/'cache_stage66_transport.py')
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
for n,h in plan['prior_frozen_artifacts'].items():assert sha(run/n)==h
assert sha(run/'stage66_rate_grid_audit.json')==plan['grid_audit_sha256']
assert sha(run/'stage66_transport_contract.json')==plan['contract_sha256']
smoke=read('stage66_transport_noiseless.json');assert smoke['plan_sha256']==progress['plan_sha256'] and smoke['worker_sha256']==progress['worker_sha256'] and smoke['passed']==2
rows=[];coordinates=set();integer_fields=('total_packet_attempts','accepted_packets','censored_packets','final_payload_error_packets','accepted_crc_payload_errors','container_byte_errors','forward_wire_bits','forward_qam_symbols','logical_feedback_events')
totals={p:{k:0 for k in integer_fields} for p in POLICIES}
for name,h in progress['case_sha256'].items():
    path=target/name;assert sha(path)==h;v=json.loads(path.read_text(encoding='utf-8'));i,t,s=v['image_index'],v['trial'],v['snr_index']
    assert name==f'snr{s}_image{i:02d}_trial{t}.json' and 0<=i<10 and 0<=t<5 and 0<=s<6 and (i,t,s) not in coordinates
    coordinates.add((i,t,s));assert v['snr_db']==plan['snr_db'][s] and v['plan_sha256']==progress['plan_sha256'] and v['worker_sha256']==progress['worker_sha256']
    source=next(a for a in plan['streams'] if a['image_index']==i);blob=(r/source['container_path']).read_bytes()
    assert sha(r/source['container_path'])==v['source_container_sha256']==source['container_sha256']
    assert sha(path.with_suffix('.npz'))==v['trace_arrays_sha256']
    with np.load(path.with_suffix('.npz')) as a:
        sent=a['transmitted_packets'];assert np.array_equal(sent,byte_packets(blob)) and len(sent)==v['packet_count']
        replay=replay_trace(sent,a['trace_received'],a['trace_crc_valid'],a['trace_active'],plan['maximum_attempts'])
        assert int(a['trace_active'].sum())==v['union_packet_observations']
        for policy,row in replay.items():
            final=a['trace_received'][np.asarray(row['packet_attempts'])-1,np.arange(len(sent))];assert np.array_equal(final,a['final_'+policy])
            received=packet_bytes(final,len(blob));row['container_byte_errors']=sum(x!=y for x,y in zip(received,blob));row['container_exact']=received==blob;row['semantic_token_errors_not_measured']=True
            row['complete_and_exact']=row['container_exact'] and row['censored_packets']==0
            assert row==v['policies'][policy]
            for key in integer_fields:totals[policy][key]+=row[key]
    assert not v['entropy_decoder_called'];rows.append(v)
assert len(rows)==progress['completed_frames'] and set(progress['case_sha256'])=={p.name for p in target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json')}
groups={}
for s,snr in enumerate(plan['snr_db']):
    subset=[v for v in rows if v['snr_index']==s]
    if len(subset)!=50:continue
    assert {(v['image_index'],v['trial']) for v in subset}=={(i,t) for i in range(10) for t in range(5)}
    policies={}
    for policy in POLICIES:
        policies[policy]=dict(frames=50,container_exact_frames=sum(v['policies'][policy]['container_exact'] for v in subset),
            complete_and_exact_frames=sum(v['policies'][policy]['complete_and_exact'] for v in subset),
            **{k:sum(v['policies'][policy][k] for v in subset) for k in integer_fields})
        policies[policy]['mean_forward_qam_symbols']=policies[policy]['forward_qam_symbols']/50
    groups[str(snr)]=policies
audit=dict(all_checks_passed=True,complete=len(rows)==300,completed_frames=len(rows),expected_frames=300,
    script_sha256=sha(__file__),plan_sha256=progress['plan_sha256'],progress_sha256=sha(run/'stage66_transport_progress.json'),
    frames_per_snr={str(snr):sum(v['snr_index']==s for v in rows) for s,snr in enumerate(plan['snr_db'])},
    complete_snr_groups=groups,exact_integer_policy_totals=totals,case_sha256=progress['case_sha256'],
    noiseless_diagnostic_frames=2,union_packet_observations=sum(v['union_packet_observations'] for v in rows),
    entropy_decoder_called=False,noisy_image_quality_evaluated=False,feedback_airtime_included=False,
    checkpoint_and_original_length_shared_out_of_band=True,paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage66_transport_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in audit.items() if k!='case_sha256'},indent=2))
