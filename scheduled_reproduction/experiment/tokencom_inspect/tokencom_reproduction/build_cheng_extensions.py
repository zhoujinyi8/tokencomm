"""Build the two unmodified upstream C++ extensions with the pinned local compiler."""
import hashlib,json,os,subprocess,sys,sysconfig
from pathlib import Path
import pybind11
r=Path(__file__).resolve().parent;work=r.parents[1];build=work/'cheng_build'
bootstrap=json.loads((r/'runs/stage65_native_bootstrap.json').read_text(encoding='utf-8'))
assert len(bootstrap)==2
source=build/'compressai-1.2.8';compiler_root=build/'llvm-mingw-20261117-ucrt-x86_64';binpath=compiler_root/'bin'
compiler=binpath/'x86_64-w64-mingw32-clang++.exe';assert compiler.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.environ['PATH']=str(binpath)+os.pathsep+os.environ['PATH']
version=subprocess.check_output([str(compiler),'--version'],text=True,encoding='utf-8')
records=[]
for name,folder in [('ans','rans'),('_CXX','ops')]:
    files=list((source/'compressai/cpp_exts'/folder).glob('*.cpp'));assert len(files)==1
    output=source/'compressai'/(name+sysconfig.get_config_var('EXT_SUFFIX'))
    args=[str(compiler),'-shared','-O3','-std=c++17',f'-I{Path(sys.base_prefix)/"include"}',f'-I{pybind11.get_include()}',
          f'-I{source/"third_party/ryg_rans"}',f'-I{source/"compressai/cpp_exts"/folder}',str(files[0]),str(Path(sys.base_prefix)/'libs/python310.lib'),'-o',str(output)]
    print('building',name,flush=True);subprocess.run(args,check=True)
    records.append(dict(module=name,source_sha256={str(p.relative_to(source)):sha(p) for p in files},binary_sha256=sha(output),output=str(output),arguments=args))
receipt=dict(compiler_version=version,compiler_sha256=sha(compiler),source_archive_sha256=bootstrap[0]['sha256'],
             source_algorithms_modified=False,python=sys.version,executable=sys.executable,pybind11_version=pybind11.__version__,extensions=records)
(r/'runs/stage65_native_build.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
