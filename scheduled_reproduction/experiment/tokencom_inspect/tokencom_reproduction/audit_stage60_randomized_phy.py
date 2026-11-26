"""Verify random packet membership, canonical masks and bounded PHY statistics."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from randomized_packetization import permutation_arrays,restore_reception
from paper_tce_reference import ideal_arq_tce,payload_tce

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
spec_path=run/'stage60_execution_spec.json';spec=read(spec_path.name)
plan=read('stage59_randomized_packetization_plan.json')
lock_path=run/'stage60_randomized_phy_lock.json';lock=read(lock_path.name)
progress=read('stage60_randomized_phy_progress.json')
assert lock['script_sha256']==spec['phy_cache_script_sha256']==sha(root/'cache_stage60_randomized_phy.py')
assert lock['plan_sha256']==progress['plan_sha256']==sha(spec_path)
assert lock['helper_sha256']==spec['packetization_helper_sha256']==sha(root/'randomized_packetization.py')
assert lock['preregistered_plan_sha256']==spec['preregistered_plan_sha256']==sha(run/'stage59_randomized_packetization_plan.json')
assert sha(run/'stage59_development_recovery_audit.json')==spec['baseline_complete_audit_sha256']
for name,digest in plan['source_token_inputs'].items():assert sha(root/name)==digest
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:images=json.loads(saved['metadata'].item())['rows']
cases=[];hashes={}
for path in sorted((run/'stage60_randomized_phy').glob('snr[0-5]_image[0-9][0-9]_trial[0-4].npz')):
    si,image,trial=map(int,re.fullmatch(r'snr([0-5])_image([0-9]{2})_trial([0-4])',path.stem).groups())
    assert image<10 and sha(path)==progress['case_sha256'][path.name]
    p,inverse=permutation_arrays(plan['permutations'][image])
    with np.load(path) as saved:
        assert saved['plan_sha256'].item()==sha(spec_path) and saved['lock_sha256'].item()==sha(lock_path)
        assert saved['source_sha256'].item()==images[image]['image_sha256']
        assert saved['permutation_sha256'].item()==plan['permutations'][image]['int64_little_endian_sha256']
        rx=saved['received'];permuted=saved['permuted_received'];oracle=saved['oracle_bad_packets'].astype(bool);crc=saved['crc_bad_packets'].astype(bool)
        assert rx.shape==permuted.shape==(256,) and rx.min()>=0 and rx.max()<1024
        true_bad=(sent[image,p]!=permuted).reshape(4,64).any(axis=0)
        assert np.array_equal(oracle,true_bad)
        canonical,oracle_mask,crc_mask=restore_reception(permuted,oracle,crc,inverse)
        assert np.array_equal(canonical,rx) and np.array_equal(oracle_mask,saved['oracle_token_mask'])
        assert np.array_equal(crc_mask,saved['crc_token_mask'])
        assert not np.any((rx!=sent[image])&~oracle_mask)
        metadata=json.loads(saved['metadata'].item())
        assert metadata['seed']==spec['phy_seed_base']+100*image+trial and metadata['snr_db']==spec['snr_db'][si]
        assert metadata['raw_bits_per_packet']==40 and metadata['crc_bits_per_packet']==51
        assert metadata['coded_bits_per_packet']==102 and metadata['qam_symbols_total']==1632
        wrong=int((rx!=sent[image]).sum())
        bit_errors=int((((sent[image].astype(np.int64)^rx.astype(np.int64))[:,None]&(1<<np.arange(10)))!=0).sum())
        assert metadata['token_errors']==wrong and metadata['bit_errors']==bit_errors
        assert metadata['oracle_bad_packets']==int(oracle.sum()) and metadata['crc_bad_packets']==int(crc.sum())
        cases.append(dict(snr_index=si,snr_db=metadata['snr_db'],image_index=image,trial=trial,
                          true_error_packets=int(oracle.sum()),crc_failed_packets=int(crc.sum()),
                          wrong_tokens=wrong,wrong_payload_bits=bit_errors,
                          crc_failed_data_correct_packets=int((crc&~oracle).sum()),
                          crc_passed_data_wrong_packets=int((~crc&oracle).sum())))
    hashes[path.name]=sha(path)
assert len(cases)==progress['completed_cases'] and set(hashes)==set(progress['case_sha256'])
coverage={};groups={}
for si,snr in enumerate(spec['snr_db']):
    subset=[c for c in cases if c['snr_index']==si];coverage[str(snr)]=len(subset)
    assert len({(c['image_index'],c['trial']) for c in subset})==len(subset)
    if len(subset)==50:
        assert {(c['image_index'],c['trial']) for c in subset}=={(i,t) for i in range(10) for t in range(5)}
        per=sum(c['true_error_packets'] for c in subset)/3200
        groups[str(snr)]=dict(complete_receptions=50,true_packet_error_rate=per,
            crc_failure_rate=sum(c['crc_failed_packets'] for c in subset)/3200,
            received_token_error_rate=sum(c['wrong_tokens'] for c in subset)/12800,
            single_shot_payload_tce=payload_tce(),ideal_arq_payload_tce=ideal_arq_tce(per))
result=dict(scope=spec['scope'],plan_sha256=sha(spec_path),preregistered_plan_sha256=spec['preregistered_plan_sha256'],
    audit_script_sha256=sha(Path(__file__)),
    lock_sha256=sha(lock_path),baseline_complete_audit_sha256=spec['baseline_complete_audit_sha256'],
    all_checks_passed=True,completed_cases=len(cases),expected_cases=300,complete=len(cases)==300,
    counts_by_snr=coverage,complete_snr_groups=groups,case_sha256=hashes,cases=cases,
    masks_in_original_token_coordinates=True,primary_holdout_changed=False,physical_retries_simulated=False)
(run/'stage60_randomized_phy_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','case_sha256')},indent=2))
