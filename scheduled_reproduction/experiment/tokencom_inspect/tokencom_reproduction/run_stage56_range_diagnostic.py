"""Execute the preregistered four-condition noiseless VQGAN development diagnostic."""
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
parser.add_argument('--max-minutes', type=float, default=10)
args = parser.parse_args()
assert 0 < args.max_minutes <= 15
deadline = time.monotonic()+60*args.max_minutes
root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
plan_path = run/'stage55_range_diagnostic_plan.json'
plan = read(plan_path.name)
metric_lock = read('stage47_metrics_lock.json')
assert sha(run/plan['source_audit']) == plan['source_audit_sha256']
assert sha(run/'stage47_metrics_lock.json') == plan['metric_lock_sha256']
assert sha(root/'pretrained/official_tokcom/last.ckpt') == plan['vqgan_weight_sha256']
data = root/'data/imagenette_stage5'
assert sha(data/'token_cache/val.pt') == plan['source_val_cache_sha256']
assert sha(run/'stage26_official_demo_tokens.npz') == plan['source_token_manifest_sha256']
assert importlib.metadata.version('lpips') == '0.1.4'
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt') == metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth') == metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth') == metric_lock['alexnet_sha256']
official = root.parents[1]/'tokencom_official_reference'
identity = {'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
            'metric_lock_sha256': sha(run/'stage47_metrics_lock.json'),
            'author_environment_sha256': sha(run/'stage46_environment.json'),
            'vq_config_sha256': sha(official/'pretrained_maskgit/VQGAN/model.yaml')}
lock_path = run/'stage56_range_execution_lock.json'
if lock_path.exists():
    assert read(lock_path.name) == identity
else:
    lock_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
sys.path.insert(0, str(official))
from Network.Taming.models.vqgan import VQModel
torch.set_num_threads(plan['torch_threads'])
cfg = OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
state = torch.load(root/'pretrained/official_tokcom/last.ckpt', map_location='cpu', weights_only=True)
assert not vq.load_state_dict(state['state_dict'], strict=False).missing_keys
del state
perceptual = lpips.LPIPS(net='alex').eval()
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu')
clip_model.eval()
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
to_pil = transforms.ToPILImage()
cache = torch.load(data/'token_cache/val.pt', map_location='cpu', weights_only=True)
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as previous:
    legacy_tokens = previous['transmitted'].copy()
target = run/'stage56_range_cases'
target.mkdir(exist_ok=True)
paths = [condition['path'] for condition in plan['conditions']]
assert paths == ['demo_input_raw_output', 'demo_input_rescaled_output',
                 'training_range_input_raw_output', 'training_range_input_rescaled_output']
budget = False
with torch.inference_mode():
    for image, source_index in enumerate(plan['selected_indices']):
        out = target/f'image{image:02d}.json'
        latent = out.with_suffix('.npz')
        if out.exists():
            saved = json.loads(out.read_text(encoding='utf-8'))
            assert saved['lock_sha256'] == sha(lock_path) and saved['arrays_sha256'] == sha(latent)
            continue
        if time.monotonic() > deadline:
            budget = True
            break
        record = cache['records'][source_index]
        source = data/record['image']
        assert sha(source) == record['sha256'] == plan['image_manifest'][image]['image_sha256']
        with Image.open(source) as opened:
            original = transform(opened.convert('RGB'))[None]
        reference = clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        tokens = {}
        raw = {}
        for name, encoder_input in [('demo', original), ('training', 2*original-1)]:
            _, _, info = vq.encode(encoder_input)
            ids = info[2].reshape(1, 16, 16)
            assert ids.min() >= 0 and ids.max() < 1024
            tokens[name] = ids.cpu().numpy().reshape(256).copy()
            raw[name] = vq.decode_code(ids)
        assert np.array_equal(tokens['demo'], legacy_tokens[image]), 'Demo tokens differ from saved source'
        outputs = [raw['demo'], (raw['demo']+1)/2, raw['training'], (raw['training']+1)/2]
        rows = []
        for name, output in zip(paths, outputs):
            mse = float((output-original).square().mean())
            distance = float(perceptual(original, output.clamp(0, 1), normalize=True).mean())
            features = clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0, 1)))[None])
            similarity = float(torch.nn.functional.cosine_similarity(reference, features).item())
            row = {'path': name, 'mse_unclipped': mse, 'psnr_raw': -10*math.log10(mse),
                   'lpips_normalized': distance, 'clip_clipped': similarity}
            assert all(math.isfinite(value) for key, value in row.items() if key != 'path')
            rows.append(row)
        temporary = latent.with_suffix('.partial.npz')
        np.savez_compressed(temporary, original=original.numpy(), demo_tokens=tokens['demo'],
                            training_tokens=tokens['training'], demo_raw=raw['demo'].numpy(),
                            training_raw=raw['training'].numpy())
        os.replace(temporary, latent)
        result = {'image_index': image, 'source_index': source_index, 'source_image': record['image'],
                  'source_sha256': sha(source), 'lock_sha256': sha(lock_path), 'arrays_sha256': sha(latent),
                  'changed_tokens': int((tokens['demo'] != tokens['training']).sum()), 'rows': rows}
        temporary = out.with_suffix('.partial.json')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        os.replace(temporary, out)
        print(f'convention image={image+1}/10 all four paths saved', flush=True)
files = sorted(target.glob('image[0-9][0-9].json'))
progress = {'plan_sha256': sha(plan_path), 'lock_sha256': sha(lock_path), 'completed_images': len(files),
            'expected_images': 10, 'complete': len(files) == 10, 'budget_reached': budget,
            'case_sha256': {path.name: sha(path) for path in files},
            'primary_holdout_changed': False, 'multisnr_plan_changed': False}
(run/'stage56_range_progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in progress.items() if key != 'case_sha256'}, indent=2))
