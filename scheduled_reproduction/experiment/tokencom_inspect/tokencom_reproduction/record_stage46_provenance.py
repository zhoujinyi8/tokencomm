"""Record implementation/environment identities without changing frozen execution."""
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
official=root.parent.parent/'tokencom_official_reference'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()
assert commit=='0422ed32dcde3347e0bb6697f343f7a9ff7b3214'
tracked_changes=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=official,text=True).splitlines()
assert not tracked_changes
versions={}
for name in ['torch','torchvision','numpy','omegaconf','Pillow','lpips','clip']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
source_files=[p for directory in ('Network','Trainer') for p in (official/directory).rglob('*.py')]
source_files.append(official/'pretrained_maskgit/VQGAN/model.yaml')
result={'python':platform.python_version(),'platform':platform.platform(),'packages':versions,
        'device':'cpu','torch_threads':4,'author_commit':commit,'tracked_author_modifications':tracked_changes,
        'author_source_sha256':{p.relative_to(official).as_posix():sha256(p) for p in source_files},
        'recovery_runner_sha256':sha256(root/'run_stage46_frozen_recovery.py'),
        'note':'provenance only; no holdout accuracy or quality metrics computed'}
out=root/'runs/stage46_environment.json'
if out.exists():assert json.loads(out.read_text(encoding='utf-8'))==result
else:out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('recorded pinned author commit and',len(source_files),'source identities')
