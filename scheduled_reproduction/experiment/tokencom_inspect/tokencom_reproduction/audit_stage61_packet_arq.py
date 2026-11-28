"""Audit observed padded ARQ traces without treating a pilot as a full cohort."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from packet_arq_accounting import POLICIES,replay_trace
from randomized_packetization import permutation_arrays

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage61_packet_arq_plan.json');lock_path=run/'stage61_packet_arq_lock_v2.json';lock=read(lock_path.name)
progress=read('stage61_packet_arq_progress.json');random=read('stage59_randomized_packetization_plan.json')
assert lock['plan_sha256']==progress['plan_sha256']==sha(run/'stage61_packet_arq_plan.json')
assert lock['helper_sha256']==sha(root/'packet_arq_accounting.py')
assert lock['script_sha256']==sha(root/'cache_stage61_packet_arq.py')
assert lock['permutation_helper_sha256']==sha(root/'randomized_packetization.py')
assert lock['component_source_sha256']==sha(root/'smoke_stage25_sionna_link.py')
assert plan['source_randomization_plan_sha256']==sha(run/'stage59_randomized_packetization_plan.json')
assert lock['tensorflow']=='2.15.1' and lock['sionna']=='0.19.2' and lock['numpy']=='1.26.4'
for name,digest in plan['source_token_inputs'].items():assert sha(root/name)==digest
contract=read('stage61_arq_contract.json');assert contract['passed']==8 and contract['failed']==0
assert contract['helper_sha256']==lock['helper_sha256']
assert contract['test_script_sha256']==sha(root/'test_packet_arq_accounting.py')
smoke=read('stage61_arq_noiseless_verification.json');assert smoke['passed']==2 and not smoke['added_channel_noise']
assert all(smoke[k]==v for k,v in lock.items())
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:images=json.loads(saved['metadata'].item())['rows']
cases=[];hashes={};coordinates=set()
for path in sorted((run/'stage61_packet_arq').glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json')):
    si,image,trial=map(int,re.fullmatch(r'snr([0-5])_image([0-9]{2})_trial([0-4])',path.stem).groups())
    assert image<10 and (si,image,trial) not in coordinates;coordinates.add((si,image,trial))
    case=json.loads(path.read_text(encoding='utf-8'));assert sha(path)==progress['case_sha256'][path.name]
    assert (case['snr_index'],case['image_index'],case['trial'])==(si,image,trial) and case['snr_db']==plan['snr_db'][si]
    assert case['plan_sha256']==lock['plan_sha256'] and case['lock_sha256']==sha(lock_path)
    assert case['source_sha256']==images[image]['image_sha256'] and sha(path.with_suffix('.npz'))==case['trace_arrays_sha256']
    p,inverse=permutation_arrays(random['permutations'][image])
    with np.load(path.with_suffix('.npz')) as trace:
        tx=trace['transmitted_packets'];assert np.array_equal(tx,sent[image,p].reshape(4,64).T)
        replayed=replay_trace(tx,trace['trace_received'],trace['trace_crc_valid'],trace['trace_active'],32)
        assert replayed==case['policies']
        assert int(trace['trace_active'].sum())==case['union_packet_observations']
        for policy,row in replayed.items():
            chosen=np.asarray(row['packet_attempts'])-1
            canonical=trace['trace_received'][chosen,np.arange(64)].T.reshape(256)[inverse]
            assert np.array_equal(canonical,trace[f'final_{policy}'])
            assert int((canonical!=sent[image]).sum())==row['final_token_errors']
            assert row['forward_wire_bits']==row['convolutional_bits']+row['padding_bits']==4*row['forward_qam_symbols']
    cases.append(case);hashes[path.name]=sha(path)
assert len(cases)==progress['completed_frame_experiments'] and set(hashes)==set(progress['case_sha256'])
fields=['total_packet_attempts','accepted_packets','censored_packets','final_payload_error_packets','final_token_errors',
        'accepted_crc_payload_errors','attempted_payload_bits','crc_bits','convolutional_bits','padding_bits',
        'forward_wire_bits','forward_qam_symbols','logical_feedback_events']
totals={policy:{key:sum(case['policies'][policy][key] for case in cases) for key in fields} for policy in POLICIES}
coverage={};groups={}
for si,snr in enumerate(plan['snr_db']):
    subset=[case for case in cases if case['snr_index']==si];coverage[str(snr)]=len(subset)
    if len(subset)==50:
        assert {(c['image_index'],c['trial']) for c in subset}=={(i,t) for i in range(10) for t in range(5)}
        groups[str(snr)]={policy:dict(payload_tce_ratio_of_sums=50*65536/sum(c['policies'][policy]['attempted_payload_bits'] for c in subset),
            forward_pixels_per_symbol_ratio_of_sums=50*65536/sum(c['policies'][policy]['forward_qam_symbols'] for c in subset),
            final_token_error_rate=sum(c['policies'][policy]['final_token_errors'] for c in subset)/12800)
            for policy in POLICIES}
pilot={(s,i,t) for i,t in plan['pilot_coordinates'] for s in range(6)}
result=dict(all_checks_passed=True,audit_script_sha256=sha(Path(__file__)),plan_sha256=lock['plan_sha256'],lock_sha256=sha(lock_path),
            completed_frame_experiments=len(cases),expected_frame_experiments=300,complete=len(cases)==300,
            pilot_complete=pilot<=coordinates,pilot_expected=24,counts_by_snr=coverage,complete_snr_groups=groups,
            observed_policy_totals=totals,union_simulated_packet_observations=sum(c['union_packet_observations'] for c in cases),
            union_observations_are_not_a_single_policy_cost=True,diagnostic_noiseless_frames=2,
            diagnostic_packet_observations=128,case_sha256=hashes,model_inference_performed=False,
            feedback_resource_cost_modeled=False,cmi_resource_cost_modeled=False,latency_modeled=False,
            old_studies_changed=False,full_paper_reproduction_complete=False)
(run/'stage61_packet_arq_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='case_sha256'},indent=2))
