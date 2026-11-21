"""Evaluate frozen recovery files with atomic per-reception metric outputs."""
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
import clip
import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs';data=root/'data/imagenette_stage44_holdout'
protocol_path=run/'stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(data/'test.jsonl')==manifest['test_jsonl_sha256']
assert importlib.metadata.version('lpips')=='0.1.4'
records=[json.loads(s) for s in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
clip_path=Path.home()/'.cache/clip/ViT-B-32.pt'
assert sha256(clip_path)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
identity={'protocol_sha256':sha256(protocol_path),'metric_script_sha256':sha256(Path(__file__)),
          'recovery_lock_sha256':sha256(run/'stage46_recovery_lock.json'),
          'clip_weight_sha256':sha256(clip_path),
          'lpips_linear_sha256':sha256(Path(lpips.__file__).parent/'weights/v0.1/alex.pth'),
          'alexnet_sha256':sha256(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')}
lock=run/'stage47_metrics_lock.json'
if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))==identity
else:lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
official=root.parent.parent/'tokencom_official_reference'
sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
torch.set_num_threads(4)
weight=root/'pretrained/official_tokcom/last.ckpt'
assert sha256(weight)==protocol['weight_sha256']['pretrained/official_tokcom/last.ckpt']
cfg=OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq=VQModel(**cfg.model.params).eval()
assert not vq.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['state_dict'],strict=False).missing_keys
perceptual=lpips.LPIPS(net='alex').eval()
clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
to_pil=transforms.ToPILImage()
target=run/'stage47_holdout_metrics';target.mkdir(exist_ok=True)
paths=['direct','oracle','crc','soft_gate_0.05','soft_gate_0.1','soft_gate_0.25']
deadline=time.monotonic()+45*60
with torch.inference_mode():
    for image,record in enumerate(records):
        source=data/record['image'];assert sha256(source)==record['sha256']
        with Image.open(source) as opened:original=transform(opened.convert('RGB'))[None]
        ref_features=clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        with np.load(run/f'stage45_holdout_tokens/image{image:03d}.npz') as saved:truth=saved['tokens'].copy()
        for trial in range(5):
            out=target/f'image{image:03d}_trial{trial}.json'
            recovery=run/f'stage46_holdout_recovery/image{image:03d}_trial{trial}.npz'
            if not recovery.exists():continue
            if out.exists():
                previous=json.loads(out.read_text(encoding='utf-8'))
                assert previous['source_recovery_sha256']==sha256(recovery)
                assert previous['metrics_lock_sha256']==sha256(lock)
                continue
            if time.monotonic()>deadline:raise SystemExit(0)
            rows=[]
            with np.load(recovery) as saved:
                assert saved['source_sha256'].item()==record['sha256']
                for name in paths:
                    ids=saved[name].copy()
                    output=vq.decode_code(torch.from_numpy(ids).long().reshape(1,16,16))
                    mse=float((output-original).square().mean())
                    distance=float(perceptual(original,output.clamp(0,1),normalize=True).mean())
                    features=clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0,1)))[None])
                    similarity=float(torch.nn.functional.cosine_similarity(ref_features,features).item())
                    rows.append({'path':name,'correct_tokens':int((ids==truth).sum()),
                                 'psnr_raw':-10*np.log10(mse),'lpips_normalized':distance,'clip_clipped':similarity})
            result={'image_index':image,'trial':trial,'source_sha256':record['sha256'],
                    'source_recovery_sha256':sha256(recovery),'metrics_lock_sha256':sha256(lock),'rows':rows}
            temporary=out.with_suffix('.partial.json')
            temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(temporary,out)
            print(f'measured image={image+1}/100 trial={trial+1}/5',flush=True)
print('saved available frozen metrics; no partial aggregate reported',flush=True)
