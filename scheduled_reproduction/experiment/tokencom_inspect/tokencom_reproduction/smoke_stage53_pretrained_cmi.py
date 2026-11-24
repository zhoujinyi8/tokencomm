"""One preselected old-development CMI interface smoke with author checkpoints."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parent
run = root / 'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan_path = run / 'stage53_development_cmi_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['script_sha256'] == sha(Path(__file__))
assert plan['scope'] == 'preselected old development case; no holdout quality evaluation'
for name, digest in plan['input_sha256'].items():
    assert sha(root / name) == digest
assert not (run / 'stage53_development_cmi_result.json').exists()
official = root.parents[1] / 'tokencom_official_reference'
sys.path.insert(0, str(official))
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT

torch.set_num_threads(4)
cfg = OmegaConf.load(official / 'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
weights = torch.load(root / 'pretrained/official_tokcom/last.ckpt', map_location='cpu', weights_only=True)
assert not vq.load_state_dict(weights['state_dict'], strict=False).missing_keys
del weights
model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024, depth=24,
                        heads=16, mlp_dim=3072, dropout=.1).eval()
weights = torch.load(root / 'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth',
                     map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
model.load_state_dict(weights, strict=True, assign=True)
del weights
wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024,
                          args=SimpleNamespace(device='cpu'))
wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
with np.load(root / plan['reception_cache']) as link:
    trial, image = plan['trial'], plan['image_index']
    tx = torch.from_numpy(link['transmitted'][image].copy()).long().reshape(1, 16, 16)
    rx = torch.from_numpy(link['received'][trial, image].copy()).long().reshape(1, 16, 16)
    mask = torch.from_numpy(link['oracle_token_mask'][trial, image].copy()).bool().reshape(1, 16, 16)
assert int(mask.sum()) == plan['expected_masked_tokens'] and mask.any()
captured = []
decode = vq.decode_code


def capture(codes):
    captured.append(codes.detach().clone())
    return decode(codes)


vq.decode_code = capture
rows = []
codes_by_path = {}
with torch.inference_mode():
    for name, mode, label in plan['calls']:
        torch.manual_seed(plan['generation_seed'])
        np.random.seed(plan['generation_seed'])
        masked = rx.clone()
        masked[mask] = 1024
        captured.clear()
        result = MaskGIT.GenDecoding(wrapper, init_code=masked, _mask=mask.clone(), orig_code=tx,
                                    nb_sample=1, labels=torch.tensor([label]), w=mode,
                                    **plan['generation_parameters'])
        assert len(captured) == 3
        final = captured[0].reshape_as(tx)
        assert torch.equal(final[~mask], rx[~mask])
        assert torch.isfinite(result[0]).all()
        ids = final.numpy().reshape(256).copy()
        codes_by_path[name] = ids
        rows.append({'path': name, 'w': mode, 'label': label, 'steps': len(result[4]),
                     'masked_tokens': int(mask.sum()), 'correct_tokens': int((final == tx).sum()),
                     'unmasked_positions_preserved': True, 'decoded_output_finite': True,
                     'token_sha256': hashlib.sha256(ids.tobytes()).hexdigest()})
        print('completed development CMI smoke:', name, flush=True)
invariant = np.array_equal(codes_by_path['without_cmi_true_placeholder'],
                           codes_by_path['without_cmi_alternate_placeholder'])
assert invariant, 'Class-dropped generation must ignore the placeholder label with a fixed sampling seed'
np.savez_compressed(run / 'stage53_development_cmi_tokens.npz', **codes_by_path)
result = {'scope': plan['scope'], 'plan_sha256': sha(plan_path), 'script_sha256': sha(Path(__file__)),
          'original_primary_protocol_changed': False, 'pretrained_weights_loaded': True,
          'current_holdout_data_accessed': False, 'image_index': image, 'trial': trial,
          'source_image_sha256': plan['source_image_sha256'], 'generation_seed': plan['generation_seed'],
          'direct_correct_tokens': int((rx == tx).sum()), 'class_dropped_label_invariance_passed': invariant,
          'rows': rows, 'caveat': 'one old development reception; interface smoke, no general CMI benefit claim'}
(run / 'stage53_development_cmi_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
