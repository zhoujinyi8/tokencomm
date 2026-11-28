"""Supplementary ideal correct-token VQGAN reference; no physical retries simulated."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes', type=float, default=8)
args = parser.parse_args()
assert 0 < args.max_minutes <= 15
deadline = time.monotonic() + args.max_minutes * 60
root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage53_ideal_vq_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['script_sha256'] == sha(Path(__file__))
for name, digest in plan['input_sha256'].items():
    assert sha(root / name) == digest
assert importlib.metadata.version('lpips') == '0.1.4'
metric_lock = json.loads((run/'stage47_metrics_lock.json').read_text(encoding='utf-8'))
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt') == metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth') == metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth') == metric_lock['alexnet_sha256']
cache_audit = json.loads((run/'stage45_cache_audit.json').read_text(encoding='utf-8'))
cache_hashes = {name.replace('\\','/'): digest for name,digest in cache_audit['cache_sha256'].items()}
official = root.parents[1]/'tokencom_official_reference'
sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
torch.set_num_threads(4)
cfg = OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
state = torch.load(root/'pretrained/official_tokcom/last.ckpt',map_location='cpu',weights_only=True)
assert not vq.load_state_dict(state['state_dict'],strict=False).missing_keys
del state
perceptual = lpips.LPIPS(net='alex').eval()
clip_model,clip_preprocess = clip.load('ViT-B/32',device='cpu')
clip_model.eval()
transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
to_pil = transforms.ToPILImage()
data = root/'data/imagenette_stage44_holdout'
records = [json.loads(line) for line in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records)==100
out_dir = run/'stage53_ideal_vq_metrics'
out_dir.mkdir(exist_ok=True)
with torch.inference_mode():
    for image,record in enumerate(records):
        out = out_dir/f'image{image:03d}.json'
        if out.exists():
            assert json.loads(out.read_text(encoding='utf-8'))['plan_sha256']==sha(plan_path)
            continue
        if time.monotonic()>deadline:
            break
        source=data/record['image']
        assert sha(source)==record['sha256']
        with Image.open(source) as opened:
            original=transform(opened.convert('RGB'))[None]
        token_path=run/f'stage45_holdout_tokens/image{image:03d}.npz'
        assert sha(token_path)==cache_hashes[token_path.relative_to(root).as_posix()]
        with np.load(token_path) as saved:
            assert saved['source_sha256'].item()==record['sha256']
            ids=torch.from_numpy(saved['tokens'].copy()).long().reshape(1,16,16)
        output=vq.decode_code(ids)
        mse=float((output-original).square().mean())
        distance=float(perceptual(original,output.clamp(0,1),normalize=True).mean())
        ref=clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        features=clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0,1)))[None])
        cosine=float(torch.nn.functional.cosine_similarity(ref,features).item())
        row={'image_index':image,'source_sha256':record['sha256'],'plan_sha256':sha(plan_path),
             'correct_tokens':256,'psnr_raw':-10*math.log10(mse),
             'lpips_normalized':distance,'clip_clipped':cosine}
        assert all(math.isfinite(row[key]) for key in ('psnr_raw','lpips_normalized','clip_clipped'))
        temp=out.with_suffix('.partial.json')
        temp.write_text(json.dumps(row,indent=2),encoding='utf-8')
        os.replace(temp,out)
        print(f'ideal correct-token VQGAN image={image+1}/100',flush=True)
files=sorted(out_dir.glob('image[0-9][0-9][0-9].json'))
rows=[json.loads(path.read_text(encoding='utf-8')) for path in files]
assert all(row['plan_sha256']==sha(plan_path) for row in rows)
summary={'scope':plan['scope'],'completed_images':len(rows),'expected_images':100,
         'complete':len(rows)==100,'plan_sha256':sha(plan_path),'original_primary_protocol_changed':False,
         'physical_retries_simulated':False,'case_sha256':{path.name:sha(path) for path in files}}
if len(rows)==100:
    assert {row['image_index'] for row in rows}==set(range(100))
    summary['means_over_images']={key:sum(row[key] for row in rows)/100
        for key in ('correct_tokens','psnr_raw','lpips_normalized','clip_clipped')}
    analytic=json.loads((run/'stage53_payload_tce_result.json').read_text(encoding='utf-8'))
    summary['analytic_ideal_arq_payload_tce']=analytic['ideal_arq_tce_from_true_per']
    summary['analytic_tce_result_sha256']=sha(run/'stage53_payload_tce_result.json')
(run/'stage53_ideal_vq_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({key:value for key,value in summary.items() if key!='case_sha256'},indent=2))
