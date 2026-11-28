"""Fetch bounded primary arXiv source archives; inspect without extracting paths."""
import hashlib,json,tarfile,urllib.request
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;target=r/'paper_sources/stage66';target.mkdir(parents=True,exist_ok=True)
results=[]
for version in (4,5):
    name=f'2502.12096v{version}';url='https://arxiv.org/src/'+name;path=target/(name+'.tar')
    try:
        if not path.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'TokenCom-reproduction-source-audit/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                content=response.read(12_000_001);assert len(content)<=12_000_000
            with tarfile.open(fileobj=__import__('io').BytesIO(content),mode='r:*') as archive:assert archive.getmembers()
            path.write_bytes(content)
        with tarfile.open(path,mode='r:*') as archive:
            tex=[]
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith('.tex'):
                    assert member.size<=5_000_000
                    data=archive.extractfile(member).read()
                    tex.append(dict(name=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        results.append(dict(version=version,url=url,status='downloaded_and_tar_verified',archive_path=str(path.relative_to(r)),
                            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,tex=tex))
    except Exception as exc:results.append(dict(version=version,url=url,status='unavailable',error_type=type(exc).__name__,error=str(exc)))
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),sources=results,source_only=True,archive_members_extracted=False,journal_equivalence_established=False)
(r/'runs/stage66_paper_source_fetch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
