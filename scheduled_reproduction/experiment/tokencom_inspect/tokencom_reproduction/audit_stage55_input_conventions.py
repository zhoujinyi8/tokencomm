"""Read-only AST audit of pinned author training/demo conventions; no model evaluation."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
official = root.parents[1] / 'tokencom_official_reference'
commit = '0422ed32dcde3347e0bb6697f343f7a9ff7b3214'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=official, text=True).strip() == commit
assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=official, text=True).strip()
# Whole-file Git blob identities independently returned by the GitHub connector.
remote_blobs = {'Trainer/vit.py': 'e902e6f81e0f604591ea2644ac2ac03161dac474',
                'TokCom_Demo.ipynb': '499852d8b6568beb2e2ced6d18130f7847b57018',
                'Performance_metrics.py': '6f9aaf5f15fd71cc0bb020f5015a4f7662c62593',
                'Network/Taming/models/vqgan.py': '9ac55afe95a6a9caf51e0e737d06d4208effc7b6'}
identities = {}
for name, expected in remote_blobs.items():
    data = (official / name).read_bytes()
    # Windows checkout CRLF bytes differ from the repository's canonical LF blob.
    canonical = data.replace(b'\r\n', b'\n')
    blob = hashlib.sha1(b'blob '+str(len(canonical)).encode()+b'\0'+canonical).hexdigest()
    assert blob == expected, name
    identities[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha': blob,
                        'remote_connector_identity_matches': True,
                        'checkout_crlf_normalized_for_git_identity': data != canonical}

trainer_tree = ast.parse((official / 'Trainer/vit.py').read_text(encoding='utf-8'))
train = next(node for node in ast.walk(trainer_tree) if isinstance(node, ast.FunctionDef) and node.name == 'train_one_epoch')
expression = ast.dump(ast.parse('2*x-1', mode='eval').body)
normalization = next(node for node in ast.walk(train) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'x' for t in node.targets)
                     and ast.dump(node.value) == expression)
encode = next(node for node in ast.walk(train) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == 'encode')
assert normalization.lineno < encode.lineno and ast.unparse(encode.args[0]) == 'x'

notebook = json.loads((official / 'TokCom_Demo.ipynb').read_text(encoding='utf-8'))
cell8 = ast.parse(''.join(notebook['cells'][8]['source']))
cell14 = ast.parse(''.join(notebook['cells'][14]['source']))
transform = next(node for node in cell8.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'transform' for t in node.targets))
assert ast.unparse(transform.value.func) == 'transforms.Compose'
transforms = [ast.unparse(node) for node in transform.value.args[0].elts]
assert transforms == ['transforms.Resize((256, 256))', 'transforms.ToTensor()']
assert not any(isinstance(node, ast.Assign) and ast.dump(node.value) == expression for node in ast.walk(cell14))
demo_encode = next(node for node in ast.walk(cell14) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'encode')
assert ast.unparse(demo_encode.args[0]) == 'x'
metric_tree = ast.parse((official / 'Performance_metrics.py').read_text(encoding='utf-8'))
lpips_function = next(node for node in metric_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_lpips')
lpips_call = next(node for node in ast.walk(lpips_function) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'loss_fn_alex')
assert not any(keyword.arg == 'normalize' for keyword in lpips_call.keywords)
vq_tree = ast.parse((official / 'Network/Taming/models/vqgan.py').read_text(encoding='utf-8'))
decode = next(node for node in ast.walk(vq_tree) if isinstance(node, ast.FunctionDef) and node.name == 'decode_code')
assert isinstance(decode.body[-1], ast.Return) and ast.unparse(decode.body[-1].value) == 'dec'

metric_script = root / 'evaluate_stage47_frozen_metrics.py'
lock = json.loads((root/'runs/stage47_metrics_lock.json').read_text(encoding='utf-8'))
assert hashlib.sha256(metric_script.read_bytes()).hexdigest() == lock['metric_script_sha256']
assert 'normalize=True' in metric_script.read_text(encoding='utf-8')
result = {'author_commit': commit, 'author_files': identities, 'all_checks_passed': True,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'training_input': {'pixel_mapping': '2*x-1 before VQGAN encoding', 'range': '[-1,1]',
                             'normalization_line': normalization.lineno, 'encode_line': encode.lineno},
          'demo_input': {'transforms': transforms, 'range': '[0,1]', 'notebook_cell': 8,
                         'encode_cell': 14, 'encode_cell_line': demo_encode.lineno},
          'demo_lpips': {'normalize_keyword_explicit': False, 'call_line': lpips_call.lineno},
          'primary_lpips': {'normalize_keyword': True, 'script_matches_frozen_lock': True},
          'decode_code': {'returns_decoder_output_without_affine_conversion': True, 'line': decode.lineno},
          'model_inference_performed': False, 'primary_protocol_changed': False, 'pilot_plan_changed': False,
          'causal_effect_on_quality_established': False,
          'next_independent_development_question': 'Predefine a separate paired VQGAN round-trip audit on old development images comparing demo input/raw output with training-range input and explicit output rescaling. Hold image geometry and metrics fixed before seeing results. Do not reinterpret or overwrite the frozen evaluation.'}
out = root / 'runs/stage55_input_convention_audit.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'author_files'}, indent=2))
