"""Evaluate preregistered masked-token compatibility, without iterative generation."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent
run = root/'runs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
plan = read('stage56_prior_probe_plan.json')
range_plan = read('stage55_range_diagnostic_plan.json')
audit = read('stage56_range_audit.json')
assert audit['all_checks_passed'] and audit['complete']
assert plan['range_plan_sha256'] == sha(run/'stage55_range_diagnostic_plan.json')
weight = root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth'
assert sha(weight) == plan['maskgit_weight_sha256']
identity = {'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(run/'stage56_prior_probe_plan.json'),
            'range_audit_sha256': sha(run/'stage56_range_audit.json'),
            'author_environment_sha256': sha(run/'stage46_environment.json'),
            'class_mapping_script_sha256': sha(root/'evaluate_stage9_pretrained_maskgit.py')}
lock_path = run/'stage56_prior_probe_lock.json'
if lock_path.exists():
    assert read(lock_path.name) == identity
else:
    lock_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
sys.path.insert(0, str(root.parents[1]/'tokencom_official_reference'))
from Network.transformer import MaskTransformer
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS
torch.set_num_threads(plan['torch_threads'])
model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024, depth=24,
                        heads=16, mlp_dim=3072, dropout=.1).eval()
state = torch.load(weight, map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
model.load_state_dict(state, strict=True, assign=True)
del state
target = run/'stage56_prior_probe_cases'
target.mkdir(exist_ok=True)
with torch.inference_mode():
    for image in range(10):
        out = target/f'image{image:02d}.json'
        arrays_path = out.with_suffix('.npz')
        if out.exists():
            previous = json.loads(out.read_text(encoding='utf-8'))
            assert previous['lock_sha256'] == sha(lock_path) and previous['arrays_sha256'] == sha(arrays_path)
            continue
        source_json = run/f'stage56_range_cases/image{image:02d}.json'
        assert sha(source_json) == audit['case_sha256'][source_json.name]
        source = json.loads(source_json.read_text(encoding='utf-8'))
        source_arrays = source_json.with_suffix('.npz')
        assert sha(source_arrays) == source['arrays_sha256']
        mask = np.sort(np.random.default_rng(plan['mask_seed_base']+image).choice(256, 128, replace=False))
        label = IMAGENET_CLASS_IDS[range_plan['image_manifest'][image]['class_name']]
        rows = []
        arrays = {'mask_indices': mask}
        with np.load(source_arrays) as saved:
            for tokenization in plan['tokenizations']:
                ids = torch.from_numpy(saved[f'{tokenization}_tokens'].copy()).long()[None]
                truth = ids[0, mask].clone()
                arrays[f'{tokenization}_target'] = truth.numpy()
                masked = ids.clone()
                masked[0, mask] = 1024
                for mode, dropped in plan['class_modes'].items():
                    name = f'{tokenization}_{mode}'
                    logits = model(masked.reshape(1, 16, 16), torch.tensor([label]),
                                   drop_label=torch.tensor([dropped]))[0, mask]
                    assert logits.shape == (128, 1025) and torch.isfinite(logits).all()
                    loss = float(torch.nn.functional.cross_entropy(logits, truth))
                    correct = int((logits.argmax(-1) == truth).sum())
                    arrays[name] = logits.numpy()
                    rows.append({'path': name, 'cross_entropy': loss, 'correct_masked_tokens': correct,
                                 'masked_tokens': 128, 'top1_accuracy': correct/128})
        temporary = arrays_path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, arrays_path)
        result = {'image_index': image, 'label': label, 'source_range_json_sha256': sha(source_json),
                  'lock_sha256': sha(lock_path), 'arrays_sha256': sha(arrays_path), 'rows': rows}
        temporary = out.with_suffix('.partial.json')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        os.replace(temporary, out)
        print(f'prior probe image={image+1}/10 four paths saved', flush=True)
files = sorted(target.glob('image[0-9][0-9].json'))
result = {'plan_sha256': identity['plan_sha256'], 'lock_sha256': sha(lock_path), 'complete_images': len(files),
          'complete': len(files) == 10, 'case_sha256': {path.name: sha(path) for path in files},
          'primary_holdout_changed': False, 'multisnr_plan_changed': False}
(run/'stage56_prior_probe_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'case_sha256'}, indent=2))
