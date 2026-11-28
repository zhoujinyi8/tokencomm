"""Restore an exact unsigned published commit locally when native fetch is unavailable."""
import argparse
import datetime
import hashlib
import itertools
import json
import subprocess
import urllib.request
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--commit',required=True)
parser.add_argument('--expected-parent',required=True)
parser.add_argument('--branch',default='reproduction/automation-30-20261122',
                    help='Exact existing local publication branch; checked before staging or ref changes')
parser.add_argument('--metadata',type=Path)
parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
assert (checkout/'.git').is_file()
receipt=(root/args.receipt).resolve();assert receipt.parent==(root/'runs').resolve()
for value in (args.commit,args.expected_parent):assert len(value)==40 and all(c in '0123456789abcdef' for c in value)
def git(*values):return subprocess.check_output(['git',*values],cwd=checkout,text=True,encoding='utf-8').strip()
assert Path(git('rev-parse','--show-toplevel')).resolve()==checkout.resolve()
head=git('rev-parse','HEAD')
assert head in (args.expected_parent,args.commit)
git('check-ref-format','--branch',args.branch)
branch=git('symbolic-ref','HEAD');assert branch=='refs/heads/'+args.branch
if args.metadata:
    meta=json.loads(args.metadata.read_text(encoding='utf-8'))
else:
    url=f'https://api.github.com/repos/zhoujinyi8/tokencomm/git/commits/{args.commit}'
    request=urllib.request.Request(url,headers={'User-Agent':'TokCom-reproduction','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=12) as response:meta=json.load(response)
assert meta['sha']==args.commit and len(meta['parents'])==1 and meta['parents'][0]['sha']==args.expected_parent
assert not meta.get('verification',{}).get('signature'),'Signed objects require their original raw representation'
extras=['.flake8','pytest.ini','requirements-ci.txt','.github/workflows/python-app.yml','conftest.py',
        'test_context_maskgit_v4.py','CI_TESTING.md','third_party/perceptual_similarity/lpips/trainer.py']
staged=[name for name in git('diff','--cached','--name-only','-z').split('\0') if name]
assert all(name.startswith('scheduled_reproduction/') or name in extras for name in staged)
subprocess.check_call(['git','-c','core.autocrlf=false','add','scheduled_reproduction',*extras],cwd=checkout)
tree=git('write-tree');assert tree==meta['tree']['sha'],'Worktree/index does not match the remote Git tree'
def person(role,zone):
    value=meta[role]
    epoch=int(datetime.datetime.fromisoformat(value['date'].replace('Z','+00:00')).timestamp())
    return f"{role} {value['name']} <{value['email']}> {epoch} {zone}\n"
zones=['+0800','+0000','-0800','-0700']+[f'{"+" if n>=0 else "-"}{abs(n)//60:02d}{abs(n)%60:02d}' for n in range(-840,841,15)]
zones=list(dict.fromkeys(zones));raw=None
for author_zone,committer_zone in itertools.product(zones,zones):
    prefix=f'tree {tree}\nparent {args.expected_parent}\n'+person('author',author_zone)+person('committer',committer_zone)
    for encoding in ('','encoding UTF-8\n','encoding utf-8\n'):
        for ending in ('','\n','\n\n','\r\n'):
            candidate=(prefix+encoding+'\n'+meta['message']+ending).encode('utf-8')
            actual=hashlib.sha1(b'commit '+str(len(candidate)).encode()+b'\0'+candidate).hexdigest()
            if actual==args.commit:raw=candidate;break
        if raw:break
    if raw:break
assert raw is not None,'No exact-byte commit match; local ref has not been moved'
actual=subprocess.check_output(['git','hash-object','-t','commit','-w','--stdin'],cwd=checkout,input=raw).decode().strip()
assert actual==args.commit
if head!=args.commit:subprocess.check_call(['git','update-ref',branch,args.commit,args.expected_parent],cwd=checkout)
tracking='refs/remotes/origin/'+args.branch
old=git('rev-parse',tracking);assert old in (args.expected_parent,args.commit)
if old!=args.commit:subprocess.check_call(['git','update-ref',tracking,args.commit,old],cwd=checkout)
result={'head_before':head,'head_after':args.commit,'tree_sha':tree,'exact_commit_object_verified':True,
        'publication_branch':args.branch,
        'tree_matches_remote_api':True,'worktree_clean':not bool(git('status','--porcelain=v1')),
        'method':'Git API metadata + exact tree and commit object identity before local ref update',
        'author_timezone_recovered':author_zone,'committer_timezone_recovered':committer_zone}
receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
