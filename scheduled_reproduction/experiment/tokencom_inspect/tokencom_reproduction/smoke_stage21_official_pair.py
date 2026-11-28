"""One dev-image forward pass through the author's paired VQGAN and MaskGIT."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch

from cache_real_tokens import load_image
from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--multistep', action='store_true')
    p.add_argument('--seed', type=int, default=20261109)
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from omegaconf import OmegaConf
    from Network.Taming.models.vqgan import VQModel
    from Network.transformer import MaskTransformer
    from Trainer.vit import MaskGIT

    torch.set_num_threads(4)
    start = time.monotonic()
    cfg = OmegaConf.load(args.official_source / 'pretrained_maskgit/VQGAN/model.yaml')
    vq = VQModel(**cfg.model.params).eval()
    vq_path = Path('pretrained/official_tokcom/last.ckpt')
    vq.load_state_dict(torch.load(vq_path, map_location='cpu', weights_only=True)['state_dict'], strict=False)
    maskgit = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024,
                             depth=24, heads=16, mlp_dim=3072, dropout=.1).eval()
    maskgit_path = Path('pretrained/official_tokcom/MaskGIT_ImageNet_256.pth')
    state = torch.load(maskgit_path, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
    maskgit.load_state_dict(state, strict=True, assign=True)
    del state
    data = Path('data/imagenette_stage5')
    cache = torch.load(data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    record = cache['records'][0]
    image_path = data / record['image']
    if sha256(image_path) != record['sha256']: raise ValueError('Image changed')
    class_name = cache['class_names'][int(cache['labels'][0])]
    class_id = IMAGENET_CLASS_IDS[class_name]
    image = load_image(image_path)[None]
    with torch.inference_mode():
        _, _, (_, _, codes) = vq.encode(image)
        original = codes.reshape(1, 256).long()
        missing = torch.arange(256).remainder(4).eq(0)
        masked = original.clone()
        masked[:, missing] = 1024
        logits = maskgit(masked.reshape(1, 16, 16), y=torch.tensor([class_id]),
                         drop_label=torch.tensor([False]))
        prediction = logits[:, :, :1024].argmax(-1)
        multistep = None
        if args.multistep:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            wrapper = SimpleNamespace(vit=maskgit, ae=vq, patch_size=16, codebook_size=1024,
                                      args=SimpleNamespace(device='cpu'))
            wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
            generated, clean, _, _, masks, rate, errors = MaskGIT.GenDecoding(
                wrapper, init_code=masked.reshape(1, 16, 16).clone(),
                _mask=missing.reshape(1, 16, 16).clone(),
                orig_code=original.reshape(1, 16, 16), nb_sample=1,
                labels=torch.tensor([class_id]), sm_temp=1.3, w=1,
                randomize='linear', r_temp=7, sched_mode='arccos', step=32)
            multistep = {
                'steps_executed': len(masks), 'unresolved_positions': int(masks[-1].sum()),
                'correct_masked_tokens': round(float(rate) * 256),
                'masked_tokens': int(errors),
                'mse_to_clean_vqgan': float((generated-clean).square().mean()),
                'generated_finite': bool(torch.isfinite(generated).all()),
            }
    result = {
        'scope': 'one dev image, deterministic 25% token mask, true ImageNet class; interface smoke only',
        'sampling_seed': args.seed if args.multistep else None,
        'source_index': 0, 'image_sha256': record['sha256'], 'class_name': class_name,
        'imagenet_class_id': class_id, 'mask_positions': list(range(0, 256, 4)),
        'masked_tokens': int(missing.sum()),
        'correct_masked_predictions': int((prediction[:, missing] == original[:, missing]).sum()),
        'logits_shape': list(logits.shape), 'logits_finite': bool(torch.isfinite(logits).all()),
        'elapsed_seconds': time.monotonic()-start,
        'official_gen_decoding': multistep,
        'vqgan_sha256': sha256(vq_path), 'maskgit_sha256': sha256(maskgit_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
