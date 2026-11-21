"""Compare fetched GitHub blob identities with the isolated published snapshot."""
import hashlib
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
raw=subprocess.check_output(['git','ls-tree','-rz','FETCH_HEAD','scheduled_reproduction','.flake8','pytest.ini',
                             'third_party/perceptual_similarity/lpips/trainer.py','requirements-ci.txt',
                             '.github/workflows/python-app.yml','conftest.py','test_context_maskgit_v4.py','CI_TESTING.md'],cwd=checkout)
count=0;mismatches=[]
for entry in raw.split(b'\0'):
    if not entry:continue
    header,name=entry.split(b'\t',1)
    content=(checkout/name.decode('utf-8')).read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    count+=1
    if actual!=header.decode('ascii').split()[2]:mismatches.append(name.decode('utf-8'))
result={'remote_files_checked':count,'mismatches':mismatches}
manifest=json.loads((checkout/'scheduled_reproduction/SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))
assert count==len(manifest)+9 and not mismatches
(root/'runs/stage46_github_snapshot_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
