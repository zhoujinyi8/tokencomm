import hashlib,json
from pathlib import Path
from test_cheng_stream_container import verify as container
from test_cheng_packet_budget import verify as budget
from test_imagenet100_manifest import verify as manifest
r=Path(__file__).resolve().parent
results={name:f() for name,f in [('container',container),('packet_budget',budget),('dataset_manifest',manifest)]}
assert [v['passed'] for v in results.values()]==[8,6,9] and all(v['failed']==0 for v in results.values())
sources=['cheng_stream_container.py','test_cheng_stream_container.py','cheng_packet_budget.py','test_cheng_packet_budget.py','imagenet100_manifest.py','test_imagenet100_manifest.py']
result=dict(new_checks_passed=23,failed=0,groups=results,source_sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in sources},
    synthetic_manifest_fixtures_are_not_dataset_evidence=True,physical_simulations_performed=0,model_inference_performed=False)
(r/'runs/stage65_contracts.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
