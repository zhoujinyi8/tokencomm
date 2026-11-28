"""Install hash-verified pinned wheels into the separate Cheng environment."""
import hashlib,json,subprocess,urllib.request
from pathlib import Path
r=Path(__file__).resolve().parent;work=r.parents[1];cache=work/'cheng_build/wheels';cache.mkdir(parents=True,exist_ok=True)
packages={'pybind11':'2.13.6','pandas':'2.2.3','pytorch-msssim':'1.0.0','torch-geometric':'2.6.1','xxhash':'3.6.0','python-dateutil':'2.9.0.post0','pytz':'2025.2','tzdata':'2025.2'}
receipts=[];wheels=[]
for name,version in packages.items():
    data=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30))
    candidates=[f for f in data['urls'] if f['filename'].endswith('py3-none-any.whl') or f['filename'].endswith('py2.py3-none-any.whl') or ('cp310-cp310-win_amd64.whl' in f['filename'])]
    assert len(candidates)==1,(name,[v['filename'] for v in candidates])
    asset=candidates[0];p=cache/asset['filename'];digest=asset['digests']['sha256']
    if not p.exists():p.write_bytes(urllib.request.urlopen(asset['url'],timeout=60).read())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    wheels.append(str(p));receipts.append(dict(package=name,version=version,filename=p.name,url=asset['url'],sha256=digest))
    print('verified',p.name,flush=True)
subprocess.run([str(work/'cheng_env/Scripts/python.exe'),'-m','pip','install','--no-index','--no-deps',*wheels],check=True)
(r/'runs/stage65_dependency_wheels.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
