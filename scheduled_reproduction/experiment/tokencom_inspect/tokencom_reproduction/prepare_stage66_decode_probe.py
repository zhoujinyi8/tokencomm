"""Freeze six fixed coordinates before any native decoding of received corrupted streams."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from cheng_byte_transport import packet_bytes
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
ta=read('stage66_transport_audit.json');tp=read('stage66_transport_plan.json');grid=read('stage66_rate_grid.json');base=read('stage66_rate_grid_plan.json')
assert ta['all_checks_passed'] and ta['complete'];arm=next(v for v in grid['arms'] if v['arm_id']==tp['selected_arm_id'])
source=next(v for v in tp['streams'] if v['image_index']==0);size=(r/source['container_path']).stat().st_size
target=run/'stage66_decode_probe';target.mkdir(exist_ok=True);cases=[]
for s,snr in enumerate(tp['snr_db']):
    name=f'snr{s}_image00_trial0.json';path=run/'stage66_transport_cases'/name;assert sha(path)==ta['case_sha256'][name]
    frame=json.loads(path.read_text(encoding='utf-8'))
    with np.load(path.with_suffix('.npz')) as a:received=packet_bytes(a['final_single_pass_padded'],size)
    output=target/f'case{s}.ch65';assert not output.exists();output.write_bytes(received)
    cases.append(dict(case_index=s,snr_db=snr,image_index=0,trial=0,policy='single_pass_padded',received_path=str(output.relative_to(r)),received_sha256=sha(output),
        child_result_path=str((target/f'case{s}_child.json').relative_to(r)),source_frame_sha256=sha(path),source_trace_sha256=sha(path.with_suffix('.npz')),
        sender_truth_container_exact=frame['policies']['single_pass_padded']['container_exact']))
image=next(v for v in base['calibration_images'] if v['image_index']==0)
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Six fixed first-image/first-trial SNR coordinates, exploratory bounded received-stream decoding, not performance curves',
    cases=cases,selection='image=0, trial=0, all six frozen SNRs, without filtering by byte errors or parser outcomes',
    checkpoint_path=arm['checkpoint_path'],checkpoint_sha256=arm['checkpoint_sha256'],original_image_path=image['path'],original_image_sha256=image['sha256'],
    transport_audit_sha256=sha(run/'stage66_transport_audit.json'),child_sha256=sha(r/'decode_stage66_received.py'),driver_sha256=sha(r/'run_stage66_decode_probe.py'),
    source_sha256={n:sha(r/n) for n in ('cheng_runtime.py','cheng_stream_container.py')},
    guards='At most 65536 input bytes, anchor q1 identifier, 256x256 pixels, latent shape 4x4; validated before spawning native child',
    child_timeout_seconds=60,timeout_action='Terminate the owned Windows child process tree; retain failure classification',
    containment='Separate process and fixed geometry/time bound, not a memory sandbox',
    failure_categories=['header_rejected','timeout','native_process_failure','decoded_finite_pixels'],failure_imputation='none; no aggregate quality over successful-only subset',
    full_paper_reproduction_complete=False,paper_baseline_ready=False)
out=run/'stage66_decode_probe_plan.json';assert not out.exists();out.write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(dict(plan_sha256=sha(out),cases=6)))
