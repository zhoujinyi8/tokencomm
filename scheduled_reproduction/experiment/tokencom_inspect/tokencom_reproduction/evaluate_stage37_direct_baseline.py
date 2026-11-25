"""Evaluate received-token direct VQGAN decoding with the fixed image metrics."""
import json
import sys
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pretrained_vq import sha256

root = Path(__file__).resolve().parent
official = root.parent.parent / 'tokencom_official_reference'
sys.path.insert(0, str(official))
from omegaconf import OmegaConf
from Network.Taming.models.vqgan import VQModel

torch.set_num_threads(4)
run = root / 'runs'
link = np.load(run / 'stage26_official_token_link_snr8_5trials.npz')
meta = json.loads(np.load(run / 'stage26_official_demo_tokens.npz')['metadata'].item())
data = root / 'data/imagenette_stage5'
cache = torch.load(data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
cfg = OmegaConf.load(official / 'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
weight = root / 'pretrained/official_tokcom/last.ckpt'
assert sha256(weight) == meta['vqgan_sha256']
assert not vq.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True)['state_dict'], strict=False).missing_keys
perceptual = lpips.LPIPS(net='alex').eval()
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu')
clip_model.eval()
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
to_pil = transforms.ToPILImage()
rows = []
with torch.inference_mode():
    for image in range(10):
        record = cache['records'][meta['selected_indices'][image]]
        path = data / record['image']
        assert sha256(path) == meta['rows'][image]['image_sha256']
        with Image.open(path) as source:
            original = preprocess(source.convert('RGB'))[None]
        original_features = clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        for trial in range(5):
            tokens = torch.from_numpy(link['received'][trial, image].copy()).long().reshape(1, 16, 16)
            output = vq.decode_code(tokens)
            mse = float((original-output).square().mean())
            perceptual_value = float(perceptual(original, output.clamp(0, 1), normalize=True).mean())
            features = clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0, 1)))[None])
            similarity = float(torch.nn.functional.cosine_similarity(original_features, features).item())
            rows.append({'trial': trial, 'image_index': image,
                         'class_name': meta['rows'][image]['class_name'],
                         'correct_tokens': int((link['received'][trial, image] == link['transmitted'][image]).sum()),
                         'psnr_raw': -10*np.log10(mse), 'lpips_normalized': perceptual_value,
                         'clip_clipped': similarity})
        print(f'DONE image={image}', flush=True)
result = {'scope': '10 development images, 5 saved AWGN 8 dB receptions; direct received-token decoding',
          'vqgan_sha256': meta['vqgan_sha256'], 'lpips_version': '0.1.4',
          'clip_weight_sha256': sha256(Path.home()/'.cache/clip/ViT-B-32.pt'), 'rows': rows}
(run / 'stage37_direct_baseline.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
