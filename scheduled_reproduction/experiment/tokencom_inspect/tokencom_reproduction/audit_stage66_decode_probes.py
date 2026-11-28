"""Verify received-byte reconstruction, bounded-probe identities and retained diagnostic outcomes."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from cheng_byte_transport import packet_bytes
from cheng_stream_container import unpack
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
transport=read('stage66_transport_plan.json');audit=read('stage66_transport_audit.json');assert audit['all_checks_passed'] and audit['complete']
summaries=[]
for prefix in ('stage66_decode_probe','stage66_crc_miss_probe'):
    plan=read(prefix+'_plan.json');result=read(prefix+'.json');assert result['all_checks_passed'] and result['plan_sha256']==sha(run/(prefix+'_plan.json'))
    driver='run_stage66_decode_probe.py' if prefix=='stage66_decode_probe' else 'probe_stage66_crc_miss.py'
    assert plan['driver_sha256']==result['driver_sha256']==sha(r/driver)
    assert plan['child_sha256']==sha(r/'decode_stage66_received.py') and sha(r/plan['checkpoint_path'])==plan['checkpoint_sha256']
    assert plan['transport_audit_sha256']==sha(run/'stage66_transport_audit.json')
    rows=result['cases'] if 'cases' in result else [result];assert len(rows)==len(plan['cases'])
    for case,row in zip(plan['cases'],rows):
        name=f"snr{transport['snr_db'].index(case['snr_db'])}_image{case['image_index']:02d}_trial{case['trial']}.json"
        frame_path=run/'stage66_transport_cases'/name;assert sha(frame_path)==case['source_frame_sha256']==audit['case_sha256'][name]
        frame=json.loads(frame_path.read_text(encoding='utf-8'));assert sha(frame_path.with_suffix('.npz'))==case['source_trace_sha256']
        source=next(v for v in transport['streams'] if v['image_index']==case['image_index']);original=(r/source['container_path']).read_bytes()
        with np.load(frame_path.with_suffix('.npz')) as arrays:received=packet_bytes(arrays['final_'+case['policy']],len(original))
        path=r/case['received_path'];assert path.read_bytes()==received and sha(path)==case['received_sha256']
        assert (received==original)==case['sender_truth_container_exact']
        assert row['case_index']==case['case_index'] and row['plan_sha256']==result['plan_sha256']
        if row['outcome']=='header_rejected':
            rejected=False
            try:
                parsed=unpack(received);rejected=not (parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256 and tuple(parsed['shape'])==(4,4))
            except (ValueError,AssertionError):rejected=True
            assert rejected and row['metrics'] is None and not row['native_child_started']
        elif row['outcome']=='decoded_finite_pixels':
            child_path=r/case['child_result_path'];assert sha(child_path)==row['child_result_sha256'];child=json.loads(child_path.read_text(encoding='utf-8'))
            assert child==row['metrics'] and child['received_sha256']==case['received_sha256'] and child['checkpoint_sha256']==plan['checkpoint_sha256']
            assert child['child_sha256']==plan['child_sha256'] and not child['encoder_called'] and math.isfinite(child['psnr_decoded'])
        else:assert row['outcome'] in ('timeout','native_process_failure') and row['metrics'] is None
    summaries.append(dict(probe=prefix,cases=len(rows),outcomes={k:sum(v['outcome']==k for v in rows) for k in ('header_rejected','timeout','native_process_failure','decoded_finite_pixels')},
                          plan_sha256=sha(run/(prefix+'_plan.json')),result_sha256=sha(run/(prefix+'.json'))))
headers=read('stage66_received_headers_summary.json');assert headers['transport_audit_sha256']==sha(run/'stage66_transport_audit.json')
assert headers['detailed_rows_sha256']==sha(run/'stage66_received_headers.json') and headers['script_sha256']==sha(r/'audit_stage66_received_headers.py')
assert all(sum(v.values())==300 for v in headers['counts'].values())
value=dict(all_checks_passed=True,script_sha256=sha(__file__),probes=summaries,received_header_summary_sha256=sha(run/'stage66_received_headers_summary.json'),
    total_diagnostic_coordinates=7,no_quality_imputation=True,no_whole_cohort_noisy_quality_claim=True,paper_baseline_ready=False)
(run/'stage66_decode_probes_audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))
