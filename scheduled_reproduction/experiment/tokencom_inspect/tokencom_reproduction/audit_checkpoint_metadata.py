"""Read only safe checkpoint metadata; never infer a preprocessing contract from performance."""
import hashlib
import json
import pickletools
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
run = root/'runs'
plan = json.loads((run/'stage56_end_to_end_convention_plan.json').read_text(encoding='utf-8'))
excluded = {'state_dict', 'model_state_dict', 'optimizer_state_dict', 'optimizer_states',
            'lr_schedulers', 'callbacks', 'scaler', 'optimizer', 'model', 'ema_state_dict'}
needles = ('normaliz', 'preprocess', 'transform', 'input_range', 'image_mean', 'image_std')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scan(value, prefix='', depth=0):
    hits = []
    if depth > 6 or not isinstance(value, dict):
        return hits
    for key, child in value.items():
        if not isinstance(key, str) or key in excluded:
            continue
        name = f'{prefix}.{key}' if prefix else key
        if any(word in key.lower() for word in needles):
            if isinstance(child, (bool, int, float)) or child is None:
                hits.append({'field': name, 'value': child})
            elif isinstance(child, str) and len(child) < 500:
                hits.append({'field': name, 'value': child})
            elif isinstance(child, (list, tuple)) and len(child) <= 6 and all(isinstance(v, (int, float)) for v in child):
                hits.append({'field': name, 'value': list(child)})
            else:
                hits.append({'field': name, 'value_type': type(child).__name__})
        if isinstance(child, dict):
            hits.extend(scan(child, name, depth+1))
    return hits


records = {}
for name, filename, expected in [('vqgan', 'last.ckpt', plan['vqgan_weight_sha256']),
                               ('maskgit', 'MaskGIT_ImageNet_256.pth', plan['maskgit_weight_sha256'])]:
    path = root/'pretrained/official_tokcom'/filename
    digest = sha(path); assert digest == expected
    if not zipfile.is_zipfile(path):
        markers = set()
        wanted = {'epoch', 'global_step', 'pytorch-lightning_version', 'hyper_parameters', 'state_dict',
                  'normalize', 'normalization', 'preprocess', 'transforms', 'input_range', 'image_mean', 'image_std'}
        with path.open('rb') as stream:
            for _ in range(4):
                for opcode, argument, position in pickletools.genops(stream):
                    if isinstance(argument, str) and argument in wanted:
                        markers.add(argument)
            header_end = stream.tell()
        records[name] = {'filename': filename, 'sha256': digest, 'file_bytes': path.stat().st_size,
            'read_method': 'legacy pickle opcode inspection only; no object deserialization or tensor allocation',
            'header_records_scanned': 4, 'header_end_offset': header_end,
            'header_metadata_key_markers': sorted(markers), 'preprocessing_metadata_hits': [],
            'metadata_values_available': False}
        continue
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert isinstance(checkpoint, dict)
    records[name] = {'filename': filename, 'sha256': digest, 'file_bytes': path.stat().st_size,
        'top_level_keys': [str(key) for key in checkpoint.keys()][:30],
        'metadata_field_types': {str(key): type(value).__name__ for key, value in checkpoint.items() if key not in excluded},
        'preprocessing_metadata_hits': scan(checkpoint), 'metadata_values_available': True,
        'read_method': 'torch.load weights_only=True, mmap=True; no network construction'}
    del checkpoint
result = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'script_sha256': sha(Path(__file__)), 'torch_version': str(torch.__version__),
    'weights_verified': True, 'read_method': 'safe metadata/header inspection only; no network construction or inference',
    'checkpoints': records, 'model_inference_performed': False, 'scientific_protocol_changed': False,
    'actual_training_preprocessing_established': False,
    'interpretation': 'Only explicit non-tensor metadata fields are inspected. Missing fields do not establish any training convention; performance comparisons cannot identify actual checkpoint training preprocessing.'}
(run/'stage58_checkpoint_metadata_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
