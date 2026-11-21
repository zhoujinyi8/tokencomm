"""Author-weight multi-step decoding on one stage5 dev image per class.

This isolates the author's VQGAN and GenDecoding from physical channel errors.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cache_real_tokens import load_image
from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--preprocess', choices=['center_crop', 'demo_resize'], default='center_crop')
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from omegaconf import OmegaConf
    from Network.Taming.models.vqgan import VQModel
    from Network.transformer import MaskTransformer
    from Trainer.vit import MaskGIT

    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.official_source / 'pretrained_maskgit/VQGAN/model.yaml')
    vq = VQModel(**cfg.model.params).eval()
    vq_path = Path('pretrained/official_tokcom/last.ckpt')
    incompatible = vq.load_state_dict(torch.load(vq_path, map_location='cpu', weights_only=True)['state_dict'], strict=False)
    assert not incompatible.missing_keys
    model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024,
                            depth=24, heads=16, mlp_dim=3072, dropout=.1).eval()
    maskgit_path = Path('pretrained/official_tokcom/MaskGIT_ImageNet_256.pth')
    state = torch.load(maskgit_path, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
    model.load_state_dict(state, strict=True, assign=True)
    del state
    wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024,
                              args=SimpleNamespace(device='cpu'))
    wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)

    data = Path('data/imagenette_stage5')
    cache = torch.load(data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    chosen = {}
    for i, label in enumerate(cache['labels'].tolist()):
        chosen.setdefault(label, i)
    assert len(chosen) == 10
    selected = [chosen[i] for i in range(10)]
    missing = torch.arange(256).remainder(4).eq(0)
    rows = []
    start = time.monotonic()
    for index in selected:
        record = cache['records'][index]
        path = data / record['image']
        if sha256(path) != record['sha256']: raise ValueError(f'Image changed: {path}')
        label_name = cache['class_names'][int(cache['labels'][index])]
        if args.preprocess == 'demo_resize':
            with Image.open(path) as source:
                image = transforms.ToTensor()(transforms.Resize((256, 256))(source.convert('RGB')))[None]
        else:
            image = load_image(path)[None]
        torch.manual_seed(args.seed + index)
        np.random.seed(args.seed + index)
        with torch.inference_mode():
            _, _, (_, _, code) = vq.encode(image)
            original = code.reshape(1, 16, 16).long()
            masked = original.clone().reshape(1, 256)
            masked[:, missing] = 1024
            output, clean, _, _, masks, rate, errors = MaskGIT.GenDecoding(
                wrapper, init_code=masked.reshape(1, 16, 16),
                _mask=missing.reshape(1, 16, 16).clone(), orig_code=original,
                nb_sample=1, labels=torch.tensor([IMAGENET_CLASS_IDS[label_name]]),
                sm_temp=1.3, w=1, randomize='linear', r_temp=7,
                sched_mode='arccos', step=32)
            row = {'source_index': index, 'class_name': label_name,
                   'image_sha256': record['sha256'], 'seed': args.seed + index,
                   'masked_tokens': int(errors), 'correct_masked_tokens': round(float(rate) * 256),
                   'steps_executed': len(masks), 'unresolved_positions': int(masks[-1].sum()),
                   'mse_to_clean_vqgan': float((output-clean).square().mean()),
                   'output_finite': bool(torch.isfinite(output).all())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'scope': '10 stage5 dev images, one per class; author weights and GenDecoding; fixed 25% token erasure, true class, no physical channel',
              'preprocess': args.preprocess,
              'seed_base': args.seed, 'selected_indices': selected,
              'vqgan_sha256': sha256(vq_path), 'maskgit_sha256': sha256(maskgit_path),
              'images': len(rows), 'total_masked': sum(r['masked_tokens'] for r in rows),
              'total_correct': sum(r['correct_masked_tokens'] for r in rows),
              'mean_mse_to_clean_vqgan': sum(r['mse_to_clean_vqgan'] for r in rows)/len(rows),
              'elapsed_seconds': time.monotonic()-start, 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'images': result['images'], 'total_correct': result['total_correct'],
                      'total_masked': result['total_masked']}), flush=True)


if __name__ == '__main__': main()
