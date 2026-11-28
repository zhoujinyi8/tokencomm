"""Probe pinned author class-conditioning interfaces without weights or datasets."""
import ast
import hashlib
import json
import math
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

root = Path(__file__).resolve().parent
official = root.parents[1] / 'tokencom_official_reference'
sys.path.insert(0, str(official))
from Network.transformer import MaskTransformer

torch.set_num_threads(1)
torch.manual_seed(20261124)
model = MaskTransformer(img_size=32, hidden_dim=32, codebook_size=8, depth=1,
                        heads=4, mlp_dim=64, dropout=0, nclass=3).eval()
tokens = torch.tensor([[[0, 1], [2, 3]]])
captured = []
hook = model.tok_emb.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
with torch.inference_mode():
    no_class_a = model(tokens, torch.tensor([0]), drop_label=torch.tensor([True]))
    no_class_b = model(tokens, torch.tensor([2]), drop_label=torch.tensor([True]))
    with_class_a = model(tokens, torch.tensor([0]), drop_label=torch.tensor([False]))
    with_class_b = model(tokens, torch.tensor([2]), drop_label=torch.tensor([False]))
hook.remove()
assert torch.equal(no_class_a, no_class_b)
assert not torch.equal(with_class_a, with_class_b)
assert [int(x[0, -1]) for x in captured] == [12, 12, 9, 11]
assert all(x.shape == (1, 4, 9) for x in (no_class_a, no_class_b, with_class_a, with_class_b))

trainer = official / 'Trainer/vit.py'
tree = ast.parse(trainer.read_text(encoding='utf-8'))
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MaskGIT')
method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'GenDecoding')
namespace = {'torch': torch, 'np': np, 'math': math, 'random': random}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(trainer), 'exec'), namespace)
gen_decoding = namespace['GenDecoding']


class ProbeViT:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, codes, labels, drop_label):
        self.calls.append({'batch': len(codes), 'labels': labels.tolist(), 'drop_label': drop_label.tolist()})
        logits = torch.full((len(codes), 4, 9), -30.)
        logits[:, :, 1] = 30.
        return logits


class ProbeDecoder:
    def __init__(self):
        self.codes = []

    def decode_code(self, codes):
        self.codes.append(codes.detach().clone())
        return torch.zeros((len(codes), 3, 2, 2))


calls = {}
return_metric_examples = {}
for mode in (0, 1, 9):
    vit = ProbeViT()
    wrapper = SimpleNamespace(vit=vit, ae=ProbeDecoder(), patch_size=2, codebook_size=8,
                              args=SimpleNamespace(device='cpu'), adap_sche=lambda step, mode: [2, 2])
    torch.manual_seed(20261124)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = gen_decoding(wrapper, init_code=torch.full((1, 2, 2), 8),
                              _mask=torch.ones((1, 2, 2), dtype=torch.bool),
                              orig_code=torch.zeros((1, 2, 2), dtype=torch.long),
                              nb_sample=1, labels=torch.tensor([0]), w=mode,
                              randomize='no', step=2)
    assert len(result[4]) == 2
    final = wrapper.ae.codes[0]
    assert torch.all(final == 1)
    assert float(result[5]) == 0.
    return_metric_examples[str(mode)] = {'all_four_generated_tokens_wrong': True,
                                        'returned_TER_af': float(result[5]),
                                        'actual_whole_image_residual_TER': 1.0}
    expected = [True] if mode == 0 else [False] if mode == 1 else [False, True]
    assert all(call['drop_label'] == expected for call in vit.calls)
    calls[str(mode)] = vit.calls

source_files = [official / 'Network/transformer.py', trainer]
result = {
    'kind': 'synthetic interface audit; no pretrained recovery result',
    'pretrained_weights_loaded': False,
    'datasets_accessed': False,
    'frozen_evaluation_changed': False,
    'tiny_model_configuration': {'img_size': 32, 'hidden_dim': 32, 'depth': 1, 'heads': 4,
                                 'codebook_size': 8, 'nclass': 3},
    'synthetic_probe_threads': 1,
    'drop_class_logits_invariant_to_label': True,
    'conditioned_logits_distinguish_labels': True,
    'observed_class_token_ids': [int(x[0, -1]) for x in captured],
    'generation_branch_calls': calls,
    'return_metric_examples': return_metric_examples,
    'returned_TER_af_semantics': 'correct masked predictions / all image tokens; not whole-image residual TER',
    'future_cmi_ablation_modes': {'without_cmi': 0, 'with_cmi': 1},
    'author_source_sha256': {p.relative_to(official).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files},
    'probe_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(root / 'runs/stage53_cmi_interface_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'generation_branch_calls'}, indent=2))
