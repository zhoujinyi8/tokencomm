"""Classify received containers structurally; never run a native decoder here."""
import hashlib,json
from pathlib import Path
import numpy as np
from cheng_byte_transport import packet_bytes
from cheng_stream_container import unpack
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
audit=read('stage66_transport_audit.json');plan=read('stage66_transport_plan.json');assert audit['all_checks_passed'] and audit['complete']
rows=[]
for name,h in audit['case_sha256'].items():
    path=run/'stage66_transport_cases'/name;assert sha(path)==h;frame=json.loads(path.read_text(encoding='utf-8'))
    source=next(v for v in plan['streams'] if v['image_index']==frame['image_index']);original=(r/source['container_path']).read_bytes();expected=unpack(original)
    with np.load(path.with_suffix('.npz')) as arrays:
        for policy,v in frame['policies'].items():
            blob=packet_bytes(arrays['final_'+policy],len(original));reason=None
            if v['censored_packets']:category='censored'
            else:
                try:
                    parsed=unpack(blob)
                    geometry_ok=all(parsed[k]==expected[k] for k in ('model','quality','height','width','shape'))
                    category=('exact_clean_reference' if blob==original else 'parseable_corrupted_stream') if geometry_ok else 'fixed_geometry_rejected'
                except (ValueError,AssertionError) as exc:category='container_parser_rejected';reason=str(exc)[:160]
            rows.append(dict(image_index=frame['image_index'],trial=frame['trial'],snr_db=frame['snr_db'],policy=policy,category=category,parser_reason=reason,
                source_frame_sha256=h,received_container_sha256=hashlib.sha256(blob).hexdigest(),received_container_bytes=len(blob),native_decoder_called=False))
assert len(rows)==900
counts={p:{c:sum(v['policy']==p and v['category']==c for v in rows) for c in ('exact_clean_reference','container_parser_rejected','fixed_geometry_rejected','parseable_corrupted_stream','censored')} for p in frame['policies']}
result=dict(all_checks_passed=True,script_sha256=sha(__file__),transport_audit_sha256=sha(run/'stage66_transport_audit.json'),counts=counts,rows=rows,
    interpretation='Post-hoc structural diagnosis only. Parser rejection is observable; exact/corrupt labels additionally use sender truth. Parseability does not prove decodability or correctness.',native_decoder_called=False)
(run/'stage66_received_headers.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
summary={k:v for k,v in result.items() if k!='rows'};summary['detailed_rows_sha256']=sha(run/'stage66_received_headers.json')
(run/'stage66_received_headers_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
