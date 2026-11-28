"""Evaluate remaining author PSNR cases with model weights loaded once."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS

root = Path(__file__).resolve().parent
deadline = time.monotonic() + 25 * 60
official = root.parent.parent / 'tokencom_official_reference'
sys.path.insert(0, str(official))
from omegaconf import OmegaConf
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT

torch.set_num_threads(4)
run = root / 'runs'
link = np.load(run / 'stage26_official_token_link_snr8_5trials.npz')
meta = json.loads(link['metadata'].item())
selected = json.loads(np.load(run / 'stage26_official_demo_tokens.npz')['metadata'].item())
data_dir = root / 'data/imagenette_stage5'
cache = torch.load(data_dir / 'token_cache/val.pt', map_location='cpu', weights_only=True)
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
images = []
for image in range(10):
    record = cache['records'][selected['selected_indices'][image]]
    path = data_dir / record['image']
    assert sha256(path) == record['sha256'] == selected['rows'][image]['image_sha256']
    with Image.open(path) as source:
        images.append(preprocess(source.convert('RGB'))[None])

vq_path = root / 'pretrained/official_tokcom/last.ckpt'
maskgit_path = root / 'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth'
cfg = OmegaConf.load(official / 'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
assert not vq.load_state_dict(torch.load(vq_path, map_location='cpu', weights_only=True)['state_dict'], strict=False).missing_keys
model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024,
                        depth=24, heads=16, mlp_dim=3072, dropout=.1).eval()
state = torch.load(maskgit_path, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
model.load_state_dict(state, strict=True, assign=True)
del state
wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024,
                          args=SimpleNamespace(device='cpu'))
wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
vq_hash = sha256(vq_path)
model_hash = sha256(maskgit_path)
captured_codes = []
original_decode_code = vq.decode_code
def capture_decode_code(codes):
    captured_codes.append(codes.detach().clone())
    return original_decode_code(codes)
vq.decode_code = capture_decode_code
for trial in range(5):
    for image in range(10):
        out = run / f'stage38_token_transitions_trial{trial}_image{image}.json'
        if out.exists(): continue
        if time.monotonic() >= deadline:
            print('TIME_BUDGET_REACHED; saved cases can be resumed', flush=True)
            sys.exit(0)
        tx = torch.from_numpy(link['transmitted'][image].copy()).long().reshape(1, 16, 16)
        rx = torch.from_numpy(link['received'][trial, image].copy()).long().reshape(1, 16, 16)
        masks = [('oracle_data_error', torch.from_numpy(link['oracle_token_mask'][trial, image].copy()).bool().reshape(1, 16, 16)),
                 ('receiver_crc_failure', torch.from_numpy(link['crc_token_mask'][trial, image].copy()).bool().reshape(1, 16, 16))]
        cls = meta['rows'][trial * 10 + image]['class_name']
        result_rows = []
        with torch.inference_mode():
            for name, mask in masks:
                torch.manual_seed(20261112)
                np.random.seed(20261112)
                masked = rx.clone()
                masked[mask] = 1024
                captured_codes.clear()
                output, clean, _, _, history, rate, errors = MaskGIT.GenDecoding(
                    wrapper, init_code=masked, _mask=mask.clone(), orig_code=tx,
                    nb_sample=1, labels=torch.tensor([IMAGENET_CLASS_IDS[cls]]),
                    sm_temp=1.3, w=1, randomize='linear', r_temp=7,
                    sched_mode='arccos', step=32)
                raw_mse = float((output - images[image]).square().mean())
                clipped_mse = float((output.clamp(0, 1) - images[image]).square().mean())
                row = {'mask_strategy': name, 'masked_tokens': int(errors),
                       'steps_executed': len(history),
                       'correct_masked_tokens': round(float(rate) * 256),
                       'correct_unmasked_received_tokens': int((tx[~mask] == rx[~mask]).sum()),
                       'correct_total_tokens': round(float(rate) * 256) + int((tx[~mask] == rx[~mask]).sum()),
                       'mse_to_clean_vqgan': float((output - clean).square().mean()),
                       'output_finite': bool(torch.isfinite(output).all()),
                       'original_psnr_db_author_raw': -10 * np.log10(raw_mse),
                       'original_psnr_db_clipped_diagnostic': -10 * np.log10(clipped_mse),
                       'output_min': float(output.min()), 'output_max': float(output.max()),
                       'output_outside_0_1_fraction': float(((output < 0) | (output > 1)).float().mean())}
                assert len(captured_codes) == 3
                final_codes = captured_codes[0].reshape_as(tx)
                assert torch.equal(captured_codes[1].reshape_as(tx), tx)
                before_correct = rx == tx
                after_correct = final_codes == tx
                assert int(after_correct.sum()) == row['correct_total_tokens']
                assert torch.equal(final_codes[~mask], rx[~mask])
                row['masked_received_correct'] = int((mask & before_correct).sum())
                row['masked_received_wrong'] = int((mask & ~before_correct).sum())
                row['wrong_to_correct'] = int((~before_correct & after_correct).sum())
                row['correct_to_wrong'] = int((before_correct & ~after_correct).sum())
                row['correct_to_correct_masked'] = int((mask & before_correct & after_correct).sum())
                row['wrong_to_wrong'] = int((~before_correct & ~after_correct).sum())
                row['direct_correct_tokens'] = int(before_correct.sum())
                row['final_token_ids'] = final_codes.reshape(-1).tolist()
                result_rows.append(row)
        result = {'scope': 'one stage5 dev image, same Sionna reception, author GenDecoding; smoke only',
                  'snr_db': 8., 'trial': trial, 'image_index': image, 'class_name': cls,
                  'seed_sampling': 20261112, 'source_vqgan_sha256': vq_hash,
                  'maskgit_sha256': model_hash, 'rows': result_rows}
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'DONE trial={trial} image={image}', flush=True)
