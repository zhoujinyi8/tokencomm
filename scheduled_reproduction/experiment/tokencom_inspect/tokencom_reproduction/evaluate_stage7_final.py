"""One-shot final evaluation of the stage-7 MaskGIT receiver on fresh holdout tokens."""
import argparse
import json
from pathlib import Path

import torch

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import WEIGHT_SHA256, sha256
from run_pretrained import spatial_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/stage7_context_maskgit_v2_full/best.pt')
    parser.add_argument('--data', default='data/imagenette_stage7_holdout')
    parser.add_argument('--out', default='runs/stage7_final_test')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if (out / 'results.json').exists(): parser.error('Final evaluation already completed')
    if out.exists() and any(path.name != 'protocol.json' for path in out.iterdir()):
        parser.error('Only a previously frozen protocol may exist in the output directory')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ck['run_config']['test_accessed'] or ck['identity']['tokenizer_sha256'] != WEIGHT_SHA256:
        raise ValueError('Checkpoint is not a verified train/validation-only model')
    if not (Path(args.checkpoint).parent / 'status.json').exists():
        raise ValueError('Training must complete before final evaluation')
    status = json.loads((Path(args.checkpoint).parent / 'status.json').read_text(encoding='utf-8'))
    if not status['completed'] or status['test_accessed']:
        raise ValueError('Training status does not permit final evaluation')
    manifest = json.loads((Path(args.data) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['count'] != 100 or manifest['prior_file_overlap'] or manifest['prior_pixel_overlap']:
        raise ValueError('Unexpected holdout manifest')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'checkpoint_sha256': sha256(args.checkpoint),
        'checkpoint_step': ck['step'],
        'holdout_manifest_sha256': sha256(Path(args.data) / 'manifest.json'),
        'tokenizer_sha256': WEIGHT_SHA256,
        'holdout_count': manifest['count'],
        'packet_tokens': 4, 'recovery_steps': 8,
        'erasure_probabilities': [.1, .3, .5],
        'channel_seeds': [20261031, 20261032, 20261033, 20261034, 20261035],
        'metric': 'exact token recovery on erased positions only',
        'baseline': 'spatial_fill with same erased positions',
    }
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Frozen protocol differs from current checkpoint or holdout')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    # The protocol is frozen on disk before opening the encoded final holdout.
    cache = Path(args.data) / 'token_cache' / 'test.pt'
    if not cache.exists():
        parser.error('Prepare the final token cache after freezing protocol; run cache_real_tokens.py --data ... --splits test')
    data = torch.load(cache, map_location='cpu', weights_only=True)
    if data['identity']['tokenizer_sha256'] != WEIGHT_SHA256 or len(data['ids']) != manifest['count']:
        raise ValueError('Holdout cache tokenizer identity or count differs')
    if data['class_names'] != manifest['class_names']:
        raise ValueError('Holdout class names differ')
    ids = data['ids']
    model = load_context_maskgit_v2(ck)
    model.eval()
    rows = []
    with torch.inference_mode():
        for per in protocol['erasure_probabilities']:
            for seed in protocol['channel_seeds']:
                with torch.random.fork_rng():
                    torch.manual_seed(seed + int(per * 100))
                    missing = erase_packets(ids, per, protocol['packet_tokens'])
                receiver = ids.masked_fill(missing, model.codes)
                predictions = []
                for start in range(0, len(ids), 8):
                    predictions.append(model.recover(receiver[start:start + 8], missing[start:start + 8],
                                                     steps=protocol['recovery_steps']))
                pred = torch.cat(predictions)
                if not torch.equal(pred[~missing], ids[~missing]):
                    raise RuntimeError('Model changed received tokens')
                spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
                missing_count = int(missing.sum())
                row = {
                    'per': per, 'seed': seed, 'missing_tokens': missing_count,
                    'model_correct': int((pred[missing] == ids[missing]).sum()),
                    'spatial_correct': int((spatial[missing] == ids[missing]).sum()),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = []
    for per in protocol['erasure_probabilities']:
        group = [row for row in rows if row['per'] == per]
        missing_count = sum(row['missing_tokens'] for row in group)
        summary.append({
            'per': per, 'missing_tokens': missing_count,
            'model_accuracy': sum(row['model_correct'] for row in group) / missing_count,
            'spatial_accuracy': sum(row['spatial_correct'] for row in group) / missing_count,
        })
    (out / 'results.json').write_text(json.dumps({'rows': rows, 'summary': summary}, indent=2), encoding='utf-8')
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
