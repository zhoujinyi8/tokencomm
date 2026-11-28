"""Matched 8 dB end-to-end conventions, with hash-checked reuse and atomic cases."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
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
parser.add_argument('--max-minutes', type=float, default=43)
args = parser.parse_args(); assert 0 < args.max_minutes <= 50
deadline = time.monotonic()+60*args.max_minutes
root = Path(__file__).resolve().parent; run = root/'runs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: json.loads((run/n).read_text(encoding='utf-8'))
plan_path = run/'stage56_end_to_end_convention_plan.json'; plan = read(plan_path.name)
phy_audit = read('stage57_comparison_phy_audit.json'); range_audit = read('stage56_range_audit.json')
old_plan = read('stage54_development_multisnr_plan.json')
reuse_audit_path = run/'stage57_reuse_source_audit.json'
if not reuse_audit_path.exists():
    shutil.copyfile(run/'stage54_development_recovery_audit.json', reuse_audit_path)
old_audit = read(reuse_audit_path.name); assert old_audit['all_checks_passed']
old_lock = read('stage54_development_recovery_lock.json'); metric_lock = read('stage47_metrics_lock.json')
assert phy_audit['complete'] and phy_audit['all_checks_passed']
assert phy_audit['plan_sha256'] == sha(plan_path)
assert old_plan['generation_seed'] == plan['generation_seed']
assert old_plan['generation_parameters'] == plan['generation_parameters']
assert old_plan['paths'] == plan['paths'] and old_plan['class_modes'] == {'without_cmi': 0, 'with_cmi': 1}
assert old_lock['metric_lock_sha256'] == sha(run/'stage47_metrics_lock.json')
assert sha(root/'pretrained/official_tokcom/last.ckpt') == plan['vqgan_weight_sha256']
assert sha(root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth') == plan['maskgit_weight_sha256']
identity = {'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
            'phy_audit_sha256': sha(run/'stage57_comparison_phy_audit.json'),
            'range_audit_sha256': sha(run/'stage56_range_audit.json'),
            'old_recovery_audit_sha256': sha(reuse_audit_path),
            'old_execution_lock_sha256': sha(run/'stage54_development_recovery_lock.json'),
            'metric_lock_sha256': sha(run/'stage47_metrics_lock.json'),
            'author_environment_sha256': sha(run/'stage46_environment.json')}
lock_path = run/'stage57_comparison_execution_lock.json'
if lock_path.exists(): assert read(lock_path.name) == identity
else: lock_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
target = run/'stage57_comparison_recovery'; target.mkdir(exist_ok=True)
for image in range(10):
    for trial in range(5):
        source = run/f'stage54_development_recovery/snr4_image{image:02d}_trial{trial}.json'
        out = target/f'demo_image{image:02d}_trial{trial}.json'
        if not source.exists() or out.exists() or source.name not in old_audit['case_sha256']: continue
        old = json.loads(source.read_text(encoding='utf-8'))
        phy = phy_audit['cases'][image*5+trial]['pipelines']['demo']
        assert sha(source) == old_audit['case_sha256'][source.name]
        assert old['lock_sha256'] == identity['old_execution_lock_sha256']
        assert old['source_phy_sha256'] == phy['phy_sha256']
        assert sha(source.with_suffix('.npz')) == old['latent_sha256']
        arrays_path = out.with_suffix('.npz'); shutil.copyfile(source.with_suffix('.npz'), arrays_path)
        case = {'pipeline': 'demo', 'image_index': image, 'trial': trial,
            'source_sha256': old['source_sha256'], 'source_phy_sha256': old['source_phy_sha256'],
            'lock_sha256': sha(lock_path), 'arrays_sha256': sha(arrays_path), 'masked_tokens': old['masked_tokens'],
            'execution_source': 'reused_stage54', 'reused_case_sha256': sha(source),
            'reused_source_case': source.name, 'rows': old['rows']}
        tmp = out.with_suffix('.partial.json'); tmp.write_text(json.dumps(case, indent=2), encoding='utf-8'); os.replace(tmp, out)
assert importlib.metadata.version('lpips') == '0.1.4'
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt') == metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth') == metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth') == metric_lock['alexnet_sha256']
official = root.parents[1]/'tokencom_official_reference'; sys.path.insert(0, str(official))
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS
torch.set_num_threads(4)
cfg = OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq = VQModel(**cfg.model.params).eval()
state = torch.load(root/'pretrained/official_tokcom/last.ckpt', map_location='cpu', weights_only=True)
assert not vq.load_state_dict(state['state_dict'], strict=False).missing_keys; del state
model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=16, mlp_dim=3072, dropout=.1).eval()
state = torch.load(root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth', map_location='cpu', weights_only=True, mmap=True)['model_state_dict']
model.load_state_dict(state, strict=True, assign=True); del state
wrapper = SimpleNamespace(vit=model, ae=vq, patch_size=16, codebook_size=1024, args=SimpleNamespace(device='cpu'))
wrapper.adap_sche = lambda step, mode: MaskGIT.adap_sche(wrapper, step, mode=mode)
perceptual = lpips.LPIPS(net='alex').eval()
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu'); clip_model.eval()
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]); to_pil = transforms.ToPILImage()
data = root/'data/imagenette_stage5'
captured = []; decode = vq.decode_code
def capture(codes):
    captured.append(codes.detach().clone()); return decode(codes)
vq.decode_code = capture
budget = False
with torch.inference_mode():
    for image in range(10):
        source_json = run/f'stage56_range_cases/image{image:02d}.json'
        source = json.loads(source_json.read_text(encoding='utf-8'))
        assert sha(source_json) == range_audit['case_sha256'][source_json.name]
        assert sha(source_json.with_suffix('.npz')) == source['arrays_sha256']
        image_path = data/source['source_image']; assert sha(image_path) == source['source_sha256']
        with Image.open(image_path) as opened: original = transform(opened.convert('RGB'))[None]
        reference = clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        with np.load(source_json.with_suffix('.npz')) as arrays:
            sent = {'demo': arrays['demo_tokens'].copy(), 'training_code_convention': arrays['training_tokens'].copy()}
        label = IMAGENET_CLASS_IDS[plan['source_image_manifest'][image]['class_name']]
        clean = {}
        for trial in range(5):
            for pipeline in ('demo', 'training_code_convention'):
                out = target/f'{pipeline}_image{image:02d}_trial{trial}.json'; arrays_path = out.with_suffix('.npz')
                if out.exists():
                    previous = json.loads(out.read_text(encoding='utf-8'))
                    assert previous['lock_sha256'] == sha(lock_path) and previous['arrays_sha256'] == sha(arrays_path)
                    continue
                if time.monotonic() > deadline: budget = True; break
                phy_path = (run/f'stage54_development_phy/snr4_image{image:02d}_trial{trial}.npz' if pipeline == 'demo'
                            else run/f'stage57_comparison_phy/image{image:02d}_trial{trial}.npz')
                assert sha(phy_path) == phy_audit['cases'][image*5+trial]['pipelines'][pipeline]['phy_sha256']
                tx = torch.from_numpy(sent[pipeline].copy()).long().reshape(1, 16, 16)
                with np.load(phy_path) as saved:
                    rx = torch.from_numpy(saved['received'].copy()).long().reshape(1, 16, 16)
                    mask = torch.from_numpy(saved['oracle_token_mask'].copy()).bool().reshape(1, 16, 16)
                def measure(name, raw, ids, steps):
                    output = raw if pipeline == 'demo' else (raw+1)/2
                    mse = float((output-original).square().mean())
                    distance = float(perceptual(original, output.clamp(0, 1), normalize=True).mean())
                    feature = clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0, 1)))[None])
                    correct = int((ids == tx).sum())
                    row = {'path': name, 'correct_tokens': correct, 'whole_image_ter': 1-correct/256, 'steps': steps,
                        'psnr_raw': -10*math.log10(mse), 'lpips_normalized': distance,
                        'clip_clipped': float(torch.nn.functional.cosine_similarity(reference, feature).item())}
                    assert all(math.isfinite(row[key]) for key in ('psnr_raw', 'lpips_normalized', 'clip_clipped'))
                    return row
                if pipeline not in clean: clean[pipeline] = measure('ideal_correct_tokens', vq.decode_code(tx), tx, 0)
                rows = [measure('direct', vq.decode_code(rx), rx, 0)]; codes = {'direct': rx.numpy().reshape(256).copy()}
                for name, mode in [('without_cmi', 0), ('with_cmi', 1)]:
                    if not mask.any(): final = rx; row = dict(rows[0], path=name)
                    else:
                        torch.manual_seed(plan['generation_seed']); np.random.seed(plan['generation_seed'])
                        masked = rx.clone(); masked[mask] = 1024; captured.clear()
                        generated = MaskGIT.GenDecoding(wrapper, init_code=masked, _mask=mask.clone(), orig_code=tx,
                            nb_sample=1, labels=torch.tensor([label]), w=mode, **plan['generation_parameters'])
                        final = captured[0].reshape_as(tx)
                        assert torch.equal(final[~mask], rx[~mask])
                        assert math.isclose(float(generated[5]), int((final[mask] == tx[mask]).sum())/256, abs_tol=1e-8)
                        row = measure(name, generated[0], final, len(generated[4]))
                    rows.append(row); codes[name] = final.numpy().reshape(256).copy()
                rows.append(dict(clean[pipeline])); codes['ideal_correct_tokens'] = tx.numpy().reshape(256).copy()
                tmp = arrays_path.with_suffix('.partial.npz'); np.savez_compressed(tmp, **codes); os.replace(tmp, arrays_path)
                case = {'pipeline': pipeline, 'image_index': image, 'trial': trial, 'source_sha256': source['source_sha256'],
                    'source_phy_sha256': sha(phy_path), 'lock_sha256': sha(lock_path), 'arrays_sha256': sha(arrays_path),
                    'masked_tokens': int(mask.sum()), 'execution_source': 'computed_stage57', 'rows': rows}
                tmp = out.with_suffix('.partial.json'); tmp.write_text(json.dumps(case, indent=2), encoding='utf-8'); os.replace(tmp, out)
                print(f'comparison {pipeline} image={image+1}/10 trial={trial+1}/5', flush=True)
            if budget: break
        if budget: break
files = sorted(target.glob('*_image[0-9][0-9]_trial[0-4].json'))
counts = {pipeline: sum(p.name.startswith(pipeline+'_') for p in files) for pipeline in ('demo', 'training_code_convention')}
reused = sum(json.loads(p.read_text(encoding='utf-8'))['execution_source'] == 'reused_stage54' for p in files)
result = {'plan_sha256': sha(plan_path), 'lock_sha256': sha(lock_path), 'counts': counts,
    'pipeline_receptions': len(files), 'reused_demo_cases': reused, 'computed_cases': len(files)-reused,
    'complete': len(files) == 100, 'budget_reached': budget, 'case_sha256': {p.name: sha(p) for p in files},
    'primary_holdout_changed': False, 'old_multisnr_study_changed': False}
(run/'stage57_comparison_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'case_sha256'}, indent=2))
