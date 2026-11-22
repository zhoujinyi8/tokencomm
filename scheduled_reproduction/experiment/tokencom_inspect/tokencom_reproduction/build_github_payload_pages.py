"""Serialize bounded GitHub tree pages without model/data weights."""
import json
import argparse
import hashlib
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--changed-only',action='store_true',help='Compare candidate bytes to isolated checkout HEAD')
args=parser.parse_args()
root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
target=checkout/'scheduled_reproduction'
pages=root.parents[1]/'github_payload_pages';pages.mkdir(exist_ok=True)
base_blobs={}
if args.changed_only:
    listing=subprocess.check_output(['git','ls-tree','-r','-z','HEAD'],cwd=checkout)
    for record in listing.split(b'\0'):
        if not record:continue
        metadata,name=record.split(b'\t',1)
        mode,kind,sha=metadata.split()
        if kind==b'blob':base_blobs[name.decode('utf-8')]=sha.decode('ascii')

groups=[];current=[];size=0
for path in sorted(list(target.rglob('*'))+[checkout/'.flake8',checkout/'pytest.ini',
                                        checkout/'third_party/perceptual_similarity/lpips/trainer.py',
                                        checkout/'requirements-ci.txt',checkout/'.github/workflows/python-app.yml',
                                        checkout/'conftest.py',checkout/'test_context_maskgit_v4.py',checkout/'CI_TESTING.md']):
    if not path.is_file():continue
    name=path.relative_to(checkout).as_posix()
    data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode('ascii')+b'\0'+data).hexdigest()
    if args.changed_only and base_blobs.get(name)==blob:continue
    entry={'path':name,'mode':'100644','type':'blob','content':data.decode('utf-8')}
    length=len(json.dumps(entry,ensure_ascii=True))
    if current and size+length>200000:groups.append(current);current=[];size=0
    assert length<450000
    current.append(entry);size+=length
if current:groups.append(current)
for i,group in enumerate(groups):(pages/f'page{i}.json').write_text(json.dumps(group,ensure_ascii=True),encoding='ascii')
print(json.dumps({'pages':len(groups),'files':sum(map(len,groups)),'changed_only':args.changed_only}))
