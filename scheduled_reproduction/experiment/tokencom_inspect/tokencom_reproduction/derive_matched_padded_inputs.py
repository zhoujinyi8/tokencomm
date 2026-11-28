"""Derive canonical single-pass model inputs from existing padded ARQ first attempts."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from randomized_packetization import permutation_arrays,restore_reception

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan_path=run/'stage61_matched_padded_recovery_plan.json';plan=read(plan_path.name)
source=read('stage61_packet_arq_audit.json');random=read('stage59_randomized_packetization_plan.json')
assert source['all_checks_passed'] and source['complete'] and source['completed_frame_experiments']==300
assert sha(run/'stage61_packet_arq_audit.json')==plan['source_arq_audit_sha256']
assert source['plan_sha256']==plan['source_arq_plan_sha256']
assert sha(run/'stage59_randomized_packetization_plan.json')==plan['source_randomization_plan_sha256']
sender=run/'stage26_official_token_link_snr8_5trials.npz'
assert sha(sender)==plan['input_sha256'][sender.relative_to(root).as_posix()]
with np.load(sender) as saved:sent=saved['transmitted'].copy()
identity=dict(plan_sha256=sha(plan_path),source_arq_audit_sha256=sha(run/'stage61_packet_arq_audit.json'),
              script_sha256=sha(Path(__file__)),helper_sha256=sha(root/'randomized_packetization.py'))
lock_path=run/'stage61_matched_padded_input_lock.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
target=run/'stage61_matched_padded_inputs';target.mkdir(exist_ok=True)
crc_unmasked_wrong=0
for name,digest in sorted(source['case_sha256'].items()):
    case_path=run/'stage61_packet_arq'/name;assert sha(case_path)==digest
    case=json.loads(case_path.read_text(encoding='utf-8'));arrays=case_path.with_suffix('.npz')
    assert sha(arrays)==case['trace_arrays_sha256'];image=case['image_index']
    p,inverse=permutation_arrays(random['permutations'][image])
    with np.load(arrays) as trace:
        packets=trace['trace_received'][0];tx=trace['transmitted_packets']
        assert np.array_equal(tx,sent[image,p].reshape(4,64).T)
        oracle=np.any(packets!=tx,axis=1);crc=~trace['trace_crc_valid'][0].astype(bool)
        rx,oracle_mask,crc_mask=restore_reception(packets.T.reshape(256),oracle,crc,inverse)
        assert np.array_equal(rx,trace['final_single_pass_padded'])
        assert not np.any((rx!=sent[image])&~oracle_mask)
        assert np.array_equal(oracle_mask[p],np.tile(oracle,4))
        assert np.array_equal(crc_mask[p],np.tile(crc,4))
    crc_unmasked_wrong+=int(((rx!=sent[image])&~crc_mask).sum())
    out=target/case_path.with_suffix('.npz').name
    if out.exists():
        with np.load(out) as saved:
            assert saved['lock_sha256'].item()==sha(lock_path)
            assert saved['source_trace_arrays_sha256'].item()==sha(arrays)
            assert np.array_equal(saved['received'],rx)
        continue
    temporary=out.with_suffix('.partial.npz')
    np.savez_compressed(temporary,transmitted=sent[image],received=rx,oracle_token_mask=oracle_mask,crc_token_mask=crc_mask,
        source_arq_case_sha256=digest,source_trace_arrays_sha256=sha(arrays),source_image_sha256=case['source_sha256'],
        plan_sha256=sha(plan_path),lock_sha256=sha(lock_path),snr_db=case['snr_db'],image_index=image,trial=case['trial'])
    os.replace(temporary,out)
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].npz'))
assert len(files)==300
audit=dict(all_checks_passed=True,completed_inputs=300,expected_inputs=300,complete=True,lock_sha256=sha(lock_path),
    **identity,counts_by_snr={str(s):sum(p.name.startswith(f'snr{i}_') for p in files) for i,s in enumerate(plan['snr_db'])},
    crc_mask_unmasked_wrong_tokens=crc_unmasked_wrong,case_sha256={p.name:sha(p) for p in files},
    new_physical_simulations=0,new_transformer_inference=0,old_studies_changed=False)
(run/'stage61_matched_padded_input_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='case_sha256'},indent=2))
