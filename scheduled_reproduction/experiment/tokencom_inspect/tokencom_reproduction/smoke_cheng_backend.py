import importlib.metadata,json,sys
from cheng_runtime import activate,sha
r,source,compressai=activate()
from compressai import ans,_CXX
cdf=_CXX.pmf_to_quantized_cdf([.2,.3,.5],16)
assert cdf[0]==0 and cdf[-1]==65536 and all(a<b for a,b in zip(cdf,cdf[1:]))
checks=['quantized CDF endpoints and monotonicity'];encoded=[]
for label,values in [('zeros',[0]*256),('alternating',[0,1]*128),('support',[0,1,2]*20),('negative',[-5,-1,0,1,2,7]),('bypass',[-1000,1000,1,0])]:
    idx=[0]*len(values);blob=ans.RansEncoder().encode_with_indexes(values,idx,[cdf],[len(cdf)],[0])
    assert ans.RansDecoder().decode_with_indexes(blob,idx,[cdf],[len(cdf)],[0])==values
    checks.append(label+' roundtrip');encoded.append(blob)
assert ans.RansEncoder().encode_with_indexes([0]*256,[0]*256,[cdf],[len(cdf)],[0])==encoded[0]
checks.append('deterministic encoding')
from compressai.zoo.image import model_architectures
assert 'cheng2020-anchor' in model_architectures and 'cheng2020-attn' in model_architectures
checks.append('both architecture classes import')
receipt=dict(passed=len(checks),failed=0,checks=checks,script_sha256=sha(__file__),runtime_helper_sha256=sha(r/'cheng_runtime.py'),
    native_build_sha256=sha(r/'runs/stage65_native_build.json'),python=sys.version,executable=sys.executable,
    versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pandas','pybind11','torch-geometric','pytorch-msssim')},
    source_python_sha256={p.relative_to(source).as_posix():sha(p) for p in (source/'compressai').rglob('*.py')},
    source_package_version=compressai.__version__,model_inference_performed=False)
(r/'runs/stage65_backend_smoke.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_python_sha256'},indent=2))
