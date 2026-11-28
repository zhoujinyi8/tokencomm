"""Freeze then isolate the sole observed CRC-accepted corrupt frame; diagnostic, not quality selection."""
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
from cheng_byte_transport import packet_bytes
from cheng_stream_container import unpack
p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'));plan_path=run/'stage66_crc_miss_probe_plan.json'
if args.prepare:
    from datetime import datetime,timezone
    audit=read('stage66_transport_audit.json');assert audit['all_checks_passed'] and audit['complete']
    tp=read('stage66_transport_plan.json');grid=read('stage66_rate_grid.json');base=read('stage66_rate_grid_plan.json')
    frames=[]
    for name,h in audit['case_sha256'].items():
        path=run/'stage66_transport_cases'/name;assert sha(path)==h;v=json.loads(path.read_text(encoding='utf-8'))
        if not v['policies']['receiver_crc_arq_padded']['container_exact']:frames.append((path,v))
    assert len(frames)==1,'This focused probe expects the single observed event in the frozen completed cohort'
    path,frame=frames[0];image=next(v for v in base['calibration_images'] if v['image_index']==frame['image_index'])
    source=next(v for v in tp['streams'] if v['image_index']==frame['image_index']);size=(r/source['container_path']).stat().st_size
    directory=run/'stage66_crc_miss_probe';directory.mkdir(exist_ok=True)
    with np.load(path.with_suffix('.npz')) as a:blob=packet_bytes(a['final_receiver_crc_arq_padded'],size)
    received=directory/'case0.ch65';assert not received.exists();received.write_bytes(blob)
    arm=next(v for v in grid['arms'] if v['arm_id']==tp['selected_arm_id'])
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Post-hoc mechanism diagnosis of ALL observed corrupted CRC-accepted frames (one frame); no performance estimation',
        cases=[dict(case_index=0,snr_db=frame['snr_db'],image_index=frame['image_index'],trial=frame['trial'],policy='receiver_crc_arq_padded',
            received_path=str(received.relative_to(r)),received_sha256=sha(received),child_result_path=str((directory/'case0_child.json').relative_to(r)),
            source_frame_path=str(path.relative_to(r)),source_frame_sha256=sha(path),source_trace_sha256=sha(path.with_suffix('.npz')),sender_truth_container_exact=False)],
        checkpoint_path=arm['checkpoint_path'],checkpoint_sha256=arm['checkpoint_sha256'],original_image_path=image['path'],original_image_sha256=image['sha256'],
        child_sha256=sha(r/'decode_stage66_received.py'),driver_sha256=sha(__file__),transport_audit_sha256=sha(run/'stage66_transport_audit.json'),
        child_timeout_seconds=60,no_failure_imputation=True,paper_baseline_ready=False)
    assert not plan_path.exists();plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(plan,indent=2))
else:
    plan=json.loads(plan_path.read_text(encoding='utf-8'));assert sha(__file__)==plan['driver_sha256'] and sha(r/'decode_stage66_received.py')==plan['child_sha256']
    assert sha(run/'stage66_transport_audit.json')==plan['transport_audit_sha256']
    case=plan['cases'][0];path=r/case['received_path'];assert sha(path)==case['received_sha256']
    assert path.stat().st_size<=65536;parsed=unpack(path.read_bytes())
    assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256 and tuple(parsed['shape'])==(4,4)
    directory=run/'stage66_crc_miss_probe';result=dict(plan_sha256=sha(plan_path),driver_sha256=sha(__file__),case_index=0,metrics=None)
    # Child processes inherit this mode: native failure dialogs must not block the bounded parent.
    if os.name=='nt':__import__('ctypes').windll.kernel32.SetErrorMode(0x0001|0x0002)
    with open(directory/'stdout.log','wb') as stdout,open(directory/'stderr.log','wb') as stderr:
        child=subprocess.Popen([str(r.parents[1]/'cheng_env/Scripts/python.exe'),str(r/'decode_stage66_received.py'),'--plan',str(plan_path),'--case','0'],cwd=r,
            stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        started=time.monotonic()
        try:code=child.wait(timeout=plan['child_timeout_seconds'])
        except subprocess.TimeoutExpired:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            else:child.kill()
            child.wait(timeout=10);result.update(outcome='timeout',exit_code=child.returncode)
        else:
            child_path=r/case['child_result_path'];result['exit_code']=code
            if code!=0 or not child_path.exists():result['outcome']='native_process_failure'
            else:
                value=json.loads(child_path.read_text(encoding='utf-8'));assert value['received_sha256']==case['received_sha256'] and value['plan_sha256']==sha(plan_path)
                assert value['checkpoint_sha256']==plan['checkpoint_sha256'] and math.isfinite(value['psnr_decoded'])
                result.update(outcome='decoded_finite_pixels',metrics=value,child_result_sha256=sha(child_path))
    result.update(elapsed_seconds=time.monotonic()-started,all_checks_passed=True,post_hoc_failure_mechanism_probe=True,no_cohort_quality_claim=True)
    (run/'stage66_crc_miss_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
