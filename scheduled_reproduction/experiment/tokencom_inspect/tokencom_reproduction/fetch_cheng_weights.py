import hashlib,json,urllib.request
from pathlib import Path
r=Path(__file__).resolve().parent;target=r/'pretrained/cheng_stock';target.mkdir(parents=True,exist_ok=True)
names=['cheng2020-anchor-1-dad2ebff.pth.tar','cheng2020_attn-mse-1-465f2b64.pth.tar']
records=[]
for name in names:
    url='https://compressai.s3.amazonaws.com/models/v1/'+name;p=target/name
    expected=('dad2ebff44e86ae986d464d286d4c3531bd0f8c038233601bb8f3ad6dc7754e7' if name==names[0] else '465f2b649c3fa5c67cf19f337306fc53efc4a91b6ac465e5506278b5fc997955')
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        bad_digest=hashlib.sha256(p.read_bytes()).hexdigest()
        p.rename(p.with_name(p.name+'.unverified-'+bad_digest[:12]))
        (r/'runs/stage65_weight_download_deviation.json').write_text(json.dumps(dict(source_url=url,unverified_sha256=bad_digest,expected_sha256=expected,action='retained separate unverified file; not used for inference'),indent=2),encoding='utf-8')
    download_url=url+('?stage65_verify=1' if name==names[1] else '')
    if not p.exists():
        temp=p.with_suffix('.partial')
        with urllib.request.urlopen(download_url,timeout=60) as response,temp.open('wb') as out:
            while block:=response.read(1024*1024):out.write(block)
        assert hashlib.sha256(temp.read_bytes()).hexdigest()==expected;temp.replace(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected
    records.append(dict(name=name,url=url,download_url=download_url,sha256=digest,bytes=p.stat().st_size,quality=1,metric='mse',paper_finetuned_checkpoint=False))
    print('verified stock weight',name,digest,flush=True)
(r/'runs/stage65_stock_weights.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
