"""Explicit runtime activation; imports remain confined to the Cheng environment."""
import hashlib,json,os,sys
from pathlib import Path
_handles=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def activate():
    root=Path(__file__).resolve().parent;work=root.parents[1]
    assert Path(sys.prefix).resolve()==(work/'cheng_env').resolve()
    build=json.loads((root/'runs/stage65_native_build.json').read_text(encoding='utf-8'))
    for item in build['extensions']:assert sha(item['output'])==item['binary_sha256']
    compiler=work/'cheng_build/llvm-mingw-20261117-ucrt-x86_64/bin'
    _handles.append(os.add_dll_directory(str(compiler)))
    source=work/'cheng_build/compressai-1.2.8';sys.path.insert(0,str(source))
    import compressai
    assert compressai.__version__=='1.2.8'
    return root,source,compressai
def load_stock(model_name,path):
    import torch
    from compressai.zoo.image import model_architectures
    from compressai.zoo.pretrained import load_pretrained
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=model_architectures[model_name].from_state_dict(load_pretrained(state)).eval()
    model.update(force=True)
    return model
