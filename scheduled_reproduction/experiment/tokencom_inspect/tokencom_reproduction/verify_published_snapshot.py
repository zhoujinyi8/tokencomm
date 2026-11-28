"""Check an exported snapshot's byte hashes and Python syntax without imports."""
import hashlib
import json
import sys
from pathlib import Path

target=Path(sys.argv[1])
manifest=json.loads((target/'SNAPSHOT_SHA256.json').read_text(encoding='utf-8'))
compiled=0
for name,digest in manifest.items():
    path=target/name
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==digest,name
    if path.suffix=='.py':
        compile(raw,str(path),'exec');compiled+=1
print(json.dumps({'files_checked':len(manifest),'python_files_compiled':compiled,'hashes_match':True}))
