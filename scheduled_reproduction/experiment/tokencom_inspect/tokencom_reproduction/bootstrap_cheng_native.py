"""Fetch pinned upstream source/toolchain, verifying archives before local extraction."""
import hashlib,json,tarfile,time,urllib.request,zipfile
from pathlib import Path
r=Path(__file__).resolve().parent;work=r.parents[1];target=work/'cheng_build';target.mkdir(exist_ok=True)
assets=[('compressai-1.2.8.tar.gz','https://files.pythonhosted.org/packages/a3/65/c6e387c10f159a15f9c0099b6656636d4e2ab4e2df1412b1755c8aee55f0/compressai-1.2.8.tar.gz','d90a923694b4b66c3e39bd90316e7f177a620787190ca7550731fc16c1bc7d5e'),
 ('llvm-mingw-20261117-ucrt-x86_64.zip','https://github.com/mstorsjo/llvm-mingw/releases/download/20261117/llvm-mingw-20261117-ucrt-x86_64.zip','4d002e330a0f703625aa5ccbc37e96f8f824413e163bfc7336774aabd2c3a412')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipts=[]
for name,url,digest in assets:
    p=target/name
    if not p.exists():
        temp=p.with_suffix(p.suffix+'.partial');start=time.monotonic();total=0
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'reproduction-build'}),timeout=30) as response,temp.open('wb') as out:
            while block:=response.read(1024*1024):
                out.write(block);total+=len(block)
                if total%(16*1024*1024)==0:print(name,total,flush=True)
                if time.monotonic()-start>240:raise TimeoutError('bounded archive download exceeded four minutes')
        assert sha(temp)==digest;temp.replace(p)
    assert sha(p)==digest
    root=target/name.removesuffix('.tar.gz').removesuffix('.zip')
    marker=target/(name+'.extracted.sha256')
    if not (root.exists() and marker.exists() and marker.read_text(encoding='ascii')==digest):
        if name.endswith('.zip'):
            with zipfile.ZipFile(p) as z:
                assert all((target/n).resolve().is_relative_to(target.resolve()) for n in z.namelist())
                z.extractall(target)
        else:
            with tarfile.open(p) as t:
                assert all((target/m.name).resolve().is_relative_to(target.resolve()) and not m.issym() and not m.islnk() for m in t.getmembers())
                t.extractall(target)
        marker.write_text(digest,encoding='ascii')
    receipts.append(dict(name=name,url=url,sha256=digest,bytes=p.stat().st_size,extracted=str(root)))
    print('verified/extracted',name,flush=True)
(r/'runs/stage65_native_bootstrap.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
