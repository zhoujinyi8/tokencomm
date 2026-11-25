"""One paired reconstruction from the same author-token Sionna reception."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--image-index', type=int, default=0)
    p.add_argument('--original-image', type=Path, default=None,
                   help='Optional original image for author-style PSNR diagnostics')
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from omegaconf import OmegaConf
    from Network.Taming.models.vqgan import VQModel
    from Network.transformer import MaskTransformer
    from Trainer.vit import MaskGIT

    torch.set_num_threads(4)
    cache = np.load('runs/stage26_official_token_link_snr8_5trials.npz')
    meta = json.loads(cache['metadata'].item())
    if not (0 <= args.trial < meta['trials'] and 0 <= args.image_index < meta['images']):
        p.error('Trial or image index out of range')
    tx = torch.from_numpy(cache['transmitted'][args.image_index].copy()).long().reshape(1, 16, 16)
    rx = torch.from_numpy(cache['received'][args.trial, args.image_index].copy()).long().reshape(1, 16, 16)
    masks = {'oracle_data_error': torch.from_numpy(cache['oracle_token_mask'][args.trial, args.image_index].copy()).bool().reshape(1, 16, 16),
             'receiver_crc_failure': torch.from_numpy(cache['crc_token_mask'][args.trial, args.image_index].copy()).bool().reshape(1, 16, 16)}
    assert meta['snr_db'] == 8.
    original = None
    if args.original_image is not None:
        from PIL import Image
        from torchvision import transforms
        selected = json.loads(np.load('runs/stage26_official_demo_tokens.npz')['metadata'].item())
        assert sha256(args.original_image) == selected['rows'][args.image_index]['image_sha256']
        with Image.open(args.original_image) as source:
            original = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.ToTensor()])(source.convert('RGB'))[None]
    cfg = OmegaConf.load(args.official_source / 'pretrained_maskgit/VQGAN/model.yaml')
    vq = VQModel(**cfg.model.params).eval()
    vq_path = Path('pretrained/official_tokcom/last.ckpt')
    assert not vq.load_state_dict(torch.load(vq_path, map_location='cpu', weights_only=True)['state_dict'], strict=False).missing_keys
    model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024,
                            depth=24, heads=16, mlp_dim=3072, dropout=.1).eval()
    maskgit_path = Path('pretrained/official_tokcom/MaskGIT_ImageNet_256.pth')
    state = torch.load(maskgit_path, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
    model.load_state_dict(state, strict=True, assign=True)
    del state
    wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024,
                              args=SimpleNamespace(device='cpu'))
    wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
    cls = meta['rows'][args.trial * meta['images'] + args.image_index]['class_name']
    rows = []
    with torch.inference_mode():
        for name, mask in masks.items():
            torch.manual_seed(20261112)
            np.random.seed(20261112)
            masked = rx.clone()
            masked[mask] = 1024
            output, clean, _, _, mask_history, rate, errors = MaskGIT.GenDecoding(
                wrapper, init_code=masked, _mask=mask.clone(), orig_code=tx,
                nb_sample=1, labels=torch.tensor([IMAGENET_CLASS_IDS[cls]]),
                sm_temp=1.3, w=1, randomize='linear', r_temp=7,
                sched_mode='arccos', step=32)
            correct_masked = round(float(rate) * 256)
            remaining_correct = int((tx[~mask] == rx[~mask]).sum())
            row = {'mask_strategy': name, 'masked_tokens': int(errors),
                   'steps_executed': len(mask_history),
                   'correct_masked_tokens': correct_masked,
                   'correct_unmasked_received_tokens': remaining_correct,
                   'correct_total_tokens': correct_masked + remaining_correct,
                   'mse_to_clean_vqgan': float((output-clean).square().mean()),
                   'output_finite': bool(torch.isfinite(output).all())}
            if original is not None:
                from torch.nn import functional as F
                raw_mse = float(F.mse_loss(original, output))
                clipped_mse = float(F.mse_loss(original, output.clamp(0, 1)))
                row.update({'original_psnr_db_author_raw': -10 * np.log10(raw_mse),
                            'original_psnr_db_clipped_diagnostic': -10 * np.log10(clipped_mse),
                            'output_min': float(output.min()), 'output_max': float(output.max()),
                            'output_outside_0_1_fraction': float(((output < 0) | (output > 1)).float().mean())})
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {'scope': 'one stage5 dev image, same Sionna reception, author GenDecoding; smoke only',
              'snr_db': 8., 'trial': args.trial, 'image_index': args.image_index, 'class_name': cls,
              'seed_sampling': 20261112, 'source_vqgan_sha256': sha256(vq_path),
              'maskgit_sha256': sha256(maskgit_path), 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
