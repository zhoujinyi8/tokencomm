"""Serialize bounded GitHub tree pages without model/data weights."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
target=checkout/'scheduled_reproduction'
pages=root.parents[1]/'github_payload_pages';pages.mkdir(exist_ok=True)
groups=[];current=[];size=0
for path in sorted(list(target.rglob('*'))+[checkout/'.flake8',checkout/'pytest.ini']):
    if not path.is_file():continue
    entry={'path':path.relative_to(checkout).as_posix(),'mode':'100644','type':'blob',
           'content':path.read_bytes().decode('utf-8')}
    length=len(json.dumps(entry,ensure_ascii=True))
    if current and size+length>200000:groups.append(current);current=[];size=0
    assert length<450000
    current.append(entry);size+=length
if current:groups.append(current)
for i,group in enumerate(groups):(pages/f'page{i}.json').write_text(json.dumps(group,ensure_ascii=True),encoding='ascii')
print(json.dumps({'pages':len(groups),'files':sum(map(len,groups))}))
