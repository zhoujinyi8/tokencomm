"""Budgeted old-development CMI ablation, separate from the frozen holdout."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import clip
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes', type=float, default=14)
args = parser.parse_args()
assert 0 < args.max_minutes <= 30
deadline = time.monotonic() + args.max_minutes * 60
root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage53_development_cmi_grid_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['script_sha256'] == sha(Path(__file__))
for name, digest in plan['input_sha256'].items():
    assert sha(root / name) == digest
official = root.parents[1] / 'tokencom_official_reference'
sys.path.insert(0, str(official))
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS

torch.set_num_threads(4)
cfg = OmegaConf.load(official / 'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
state = torch.load(root / 'pretrained/official_tokcom/last.ckpt', map_location='cpu', weights_only=True)
assert not vq.load_state_dict(state['state_dict'], strict=False).missing_keys
del state
model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024, depth=24,
                        heads=16, mlp_dim=3072, dropout=.1).eval()
state = torch.load(root / 'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth',
                   map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
model.load_state_dict(state, strict=True, assign=True)
del state
wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024,
                          args=SimpleNamespace(device='cpu'))
wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
perceptual = lpips.LPIPS(net='alex').eval()
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu')
clip_model.eval()
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
to_pil = transforms.ToPILImage()
with np.load(run / 'stage26_official_token_link_snr8_5trials.npz') as link:
    link_data = {name: link[name].copy() for name in link.files if name != 'metadata'}
with np.load(run / 'stage26_official_demo_tokens.npz') as source:
    selected = json.loads(source['metadata'].item())
data_dir = root / 'data/imagenette_stage5'
cache = torch.load(data_dir / 'token_cache/val.pt', map_location='cpu', weights_only=True)
out_dir = run / 'stage53_development_cmi_grid'
out_dir.mkdir(exist_ok=True)
captured = []
decode = vq.decode_code


def capture(codes):
    captured.append(codes.detach().clone())
    return decode(codes)


vq.decode_code = capture
budget_reached = False
with torch.inference_mode():
    for image in range(10):
        record = cache['records'][selected['selected_indices'][image]]
        source = data_dir / record['image']
        assert sha(source) == record['sha256'] == selected['rows'][image]['image_sha256']
        with Image.open(source) as opened:
            original = transform(opened.convert('RGB'))[None]
        reference = clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        label = IMAGENET_CLASS_IDS[selected['rows'][image]['class_name']]
        tx = torch.from_numpy(link_data['transmitted'][image].copy()).long().reshape(1, 16, 16)
        for trial in range(5):
            out = out_dir / f'image{image:02d}_trial{trial}.json'
            if out.exists():
                saved = json.loads(out.read_text(encoding='utf-8'))
                assert saved['plan_sha256'] == sha(plan_path)
                continue
            if time.monotonic() > deadline:
                budget_reached = True
                break
            rx = torch.from_numpy(link_data['received'][trial, image].copy()).long().reshape(1, 16, 16)
            mask = torch.from_numpy(link_data['oracle_token_mask'][trial, image].copy()).bool().reshape(1, 16, 16)
            direct_output = vq.decode_code(rx)
            rows = []
            for name, mode in [('direct', None), ('without_cmi', 0), ('with_cmi', 1)]:
                if mode is None or not mask.any():
                    output, final = direct_output, rx
                    steps = 0
                else:
                    torch.manual_seed(plan['generation_seed'])
                    np.random.seed(plan['generation_seed'])
                    masked = rx.clone()
                    masked[mask] = 1024
                    captured.clear()
                    generated = MaskGIT.GenDecoding(wrapper, init_code=masked, _mask=mask.clone(),
                        orig_code=tx, nb_sample=1, labels=torch.tensor([label]), w=mode,
                        **plan['generation_parameters'])
                    output, final, steps = generated[0], captured[0].reshape_as(tx), len(generated[4])
                    assert torch.equal(final[~mask], rx[~mask])
                mse = float((output-original).square().mean())
                distance = float(perceptual(original, output.clamp(0, 1), normalize=True).mean())
                features = clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0, 1)))[None])
                cosine = float(torch.nn.functional.cosine_similarity(reference, features).item())
                row = {'path': name, 'correct_tokens': int((final == tx).sum()), 'steps': steps,
                       'psnr_raw': -10*math.log10(mse), 'lpips_normalized': distance, 'clip_clipped': cosine}
                assert all(math.isfinite(row[key]) for key in ('psnr_raw', 'lpips_normalized', 'clip_clipped'))
                rows.append(row)
            result = {'image_index': image, 'trial': trial, 'source_sha256': record['sha256'],
                      'plan_sha256': sha(plan_path), 'masked_tokens': int(mask.sum()), 'rows': rows}
            temporary = out.with_suffix('.partial.json')
            temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
            os.replace(temporary, out)
            print(f'development CMI measured image={image+1}/10 trial={trial+1}/5', flush=True)
        if budget_reached:
            break
cases = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out_dir.glob('image[0-9][0-9]_trial[0-4].json'))]
assert all(case['plan_sha256'] == sha(plan_path) for case in cases)
summary = {'scope': plan['scope'], 'completed_cases': len(cases), 'expected_cases': 50,
           'complete': len(cases) == 50, 'plan_sha256': sha(plan_path), 'primary_holdout_changed': False,
           'budget_reached': budget_reached, 'case_sha256': {p.name: sha(p) for p in out_dir.glob('image[0-9][0-9]_trial[0-4].json')}}
if len(cases) == 50:
    assert {(case['image_index'], case['trial']) for case in cases} == {(i, t) for i in range(10) for t in range(5)}
    paths = ['direct', 'without_cmi', 'with_cmi']
    keys = ['correct_tokens', 'psnr_raw', 'lpips_normalized', 'clip_clipped']
    per_image = [{name: {key: sum(next(row for row in case['rows'] if row['path']==name)[key]
                    for case in cases if case['image_index']==image)/5 for key in keys} for name in paths}
                 for image in range(10)]
    summary['means_over_images'] = {name: {key: sum(row[name][key] for row in per_image)/10 for key in keys} for name in paths}
    summary['difference_with_minus_without_cmi'] = {key: summary['means_over_images']['with_cmi'][key]-summary['means_over_images']['without_cmi'][key] for key in keys}
out = run / 'stage53_development_cmi_grid_summary.json'
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != 'case_sha256'}, indent=2))
