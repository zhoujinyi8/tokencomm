"""Native decode isolation and explicit failure classification for six frozen received streams."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from cheng_stream_container import unpack
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan_path=run/'stage66_decode_probe_plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
assert sha(__file__)==plan['driver_sha256'] and sha(r/'decode_stage66_received.py')==plan['child_sha256']
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
assert sha(run/'stage66_transport_audit.json')==plan['transport_audit_sha256']
python=r.parents[1]/'cheng_env/Scripts/python.exe';rows=[]
for case in plan['cases']:
    path=r/case['received_path'];assert sha(path)==case['received_sha256']
    result_path=run/'stage66_decode_probe'/f'case{case["case_index"]}_result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'));assert result['plan_sha256']==sha(plan_path);rows.append(result);continue
    row=dict(case_index=case['case_index'],snr_db=case['snr_db'],plan_sha256=sha(plan_path),received_sha256=case['received_sha256'],
        sender_truth_container_exact=case['sender_truth_container_exact'],native_child_started=False,metrics=None)
    try:
        assert path.stat().st_size<=65536
        parsed=unpack(path.read_bytes());assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256 and tuple(parsed['shape'])==(4,4)
    except (ValueError,AssertionError) as exc:row.update(outcome='header_rejected',reason=str(exc)[:200])
    else:
        prefix=run/'stage66_decode_probe'/f'case{case["case_index"]}';started=time.monotonic()
        with open(str(prefix)+'.stdout.log','wb') as stdout,open(str(prefix)+'.stderr.log','wb') as stderr:
            child=subprocess.Popen([str(python),str(r/'decode_stage66_received.py'),'--plan',str(plan_path),'--case',str(case['case_index'])],cwd=r,
                stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            row['native_child_started']=True
            try:code=child.wait(timeout=plan['child_timeout_seconds'])
            except subprocess.TimeoutExpired:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                else:child.kill()
                child.wait(timeout=10);row.update(outcome='timeout',exit_code=child.returncode)
            else:
                row['exit_code']=code
                child_result=r/case['child_result_path']
                if code!=0 or not child_result.exists():row['outcome']='native_process_failure'
                else:
                    decoded=json.loads(child_result.read_text(encoding='utf-8'));assert decoded['received_sha256']==case['received_sha256'] and decoded['plan_sha256']==sha(plan_path)
                    assert decoded['checkpoint_sha256']==plan['checkpoint_sha256'] and decoded['child_sha256']==plan['child_sha256']
                    row.update(outcome='decoded_finite_pixels',metrics=decoded,child_result_sha256=sha(child_result))
        row['elapsed_seconds']=time.monotonic()-started
    result_path.write_text(json.dumps(row,indent=2),encoding='utf-8');rows.append(row);print(json.dumps(row),flush=True)
assert len(rows)==6 and {v['case_index'] for v in rows}==set(range(6))
result=dict(all_checks_passed=True,plan_sha256=sha(plan_path),driver_sha256=sha(__file__),cases=rows,
    outcomes={k:sum(v['outcome']==k for v in rows) for k in plan['failure_categories']},
    native_child_processes_started=sum(v['native_child_started'] for v in rows),aggregate_quality_means=None,
    no_failure_imputation=True,exploratory_diagnostic_only=True,paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage66_decode_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
