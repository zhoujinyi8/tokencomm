"""Encode the fixed stage44 holdout, with per-image atomic resumable outputs."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs'
protocol_path=run/'stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
folder=root/'data/imagenette_stage44_holdout'
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(folder/'test.jsonl')==manifest['test_jsonl_sha256']
records=[json.loads(s) for s in (folder/'test.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records)==100
lock=run/'stage45_encoder_lock.json'
identity={'protocol_sha256':sha256(protocol_path),'manifest_sha256':sha256(folder/'test.jsonl'),
          'encoder_script_sha256':sha256(Path(__file__)),'phase':'token encoding only; no selection by outcome'}
if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))==identity
else:lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
official=root.parent.parent/'tokencom_official_reference'
sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
torch.set_num_threads(4)
weight=root/'pretrained/official_tokcom/last.ckpt'
assert sha256(weight)==protocol['weight_sha256']['pretrained/official_tokcom/last.ckpt']
cfg=OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
model=VQModel(**cfg.model.params).eval()
assert not model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['state_dict'],strict=False).missing_keys
preprocess=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
target=run/'stage45_holdout_tokens';target.mkdir(exist_ok=True)
deadline=time.monotonic()+45*60
with torch.inference_mode():
    for index,record in enumerate(records):
        out=target/f'image{index:03d}.npz'
        if out.exists():
            with np.load(out) as saved:assert saved['source_sha256'].item()==record['sha256'] and saved['tokens'].shape==(256,)
            continue
        if time.monotonic()>deadline:break
        path=folder/record['image'];assert sha256(path)==record['sha256']
        with Image.open(path) as source:image=preprocess(source.convert('RGB'))[None]
        _,_,(_,_,code)=model.encode(image)
        tokens=code.reshape(256).numpy().astype(np.int32)
        assert tokens.min()>=0 and tokens.max()<1024
        temporary=out.with_suffix('.partial.npz')
        np.savez_compressed(temporary,tokens=tokens,source_sha256=record['sha256'],label=record['label'],image_index=index)
        os.replace(temporary,out)
        print(f'encoded {index+1}/100',flush=True)
print('saved',len(list(target.glob('image[0-9][0-9][0-9].npz'))),flush=True)
