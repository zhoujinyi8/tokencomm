"""Record actual fail-closed selector contract execution, separate from model results."""
import hashlib,json
from pathlib import Path
from test_cheng_rate_selection import verify
from test_cheng_byte_transport import verify as verify_transport
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
value=verify();assert value['passed']==9 and value['failed']==0
transport=verify_transport();assert transport['passed']==8 and transport['failed']==0
value.update(script_sha256=sha(__file__),helper_sha256=sha(r/'cheng_rate_selection.py'),test_sha256=sha(r/'test_cheng_rate_selection.py'),
             byte_transport_checks=transport,transport_source_sha256={n:sha(r/n) for n in ('cheng_byte_transport.py','test_cheng_byte_transport.py')},
             new_pure_checks=17,prior_pure_ci_checks=43,total_pure_ci_checks=60,model_inference_performed=False)
(r/'runs/stage66_contracts.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))
