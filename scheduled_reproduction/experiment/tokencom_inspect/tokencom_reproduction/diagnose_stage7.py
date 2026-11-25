"""Diagnose stage-7 recovery on development validation tokens only."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import sha256
from run_pretrained import spatial_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/stage7_context_maskgit_v2_full/best.pt')
    parser.add_argument('--data', default='data/imagenette_stage5/token_cache')
    parser.add_argument('--out', default='runs/stage8_diagnostics')
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--per', type=float, default=.3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): parser.error('Use an empty output directory')
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    train = torch.load(Path(args.data) / 'train.pt', map_location='cpu', weights_only=True)
    val = torch.load(Path(args.data) / 'val.pt', map_location='cpu', weights_only=True)
    if checkpoint['identity'] != train['identity'] or train['identity'] != val['identity']:
        raise ValueError('Tokenizer or source identity mismatch')
    ids = val['ids']
    model = load_context_maskgit_v2(checkpoint)
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, args.per, 4)
        donor_indices = torch.randperm(len(ids))
    # Most frequent token from the training cache, fitted without validation labels.
    counts = torch.bincount(train['ids'].flatten(), minlength=model.codes)
    frequent_token = int(counts.argmax())
    totals = {
        'true': 0, 'one_pass': 0, 'two_steps': 0, 'four_steps': 0,
        'eight_steps': 0, 'spatial': 0, 'frequency': 0,
        'donor_one_pass': 0, 'no_context_one_pass': 0,
    }
    ce = {'correct_context': 0., 'donor_context': 0., 'no_context': 0.}
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
    with torch.inference_mode():
        for begin in range(0, len(ids), 8):
            target = ids[begin:begin + 8]
            mask = missing[begin:begin + 8]
            donor = ids[donor_indices[begin:begin + 8]]
            received = target.masked_fill(mask, model.codes)
            wrong = donor.masked_fill(mask, model.codes)
            blank = torch.full_like(target, model.codes)
            pred = {}
            for name, input_ids in [('correct_context', received), ('donor_context', wrong), ('no_context', blank)]:
                logits = model(input_ids)
                ce[name] += float(F.cross_entropy(logits[mask], target[mask], reduction='sum'))
                pred[name] = logits.argmax(-1)
            for steps, key in [(2, 'two_steps'), (4, 'four_steps'), (8, 'eight_steps')]:
                completed = model.recover(received, mask, steps=steps)
                if not torch.equal(completed[~mask], target[~mask]):
                    raise RuntimeError('Receiver changed known token')
                totals[key] += int((completed[mask] == target[mask]).sum())
            totals['true'] += int(mask.sum())
            totals['one_pass'] += int((pred['correct_context'][mask] == target[mask]).sum())
            totals['donor_one_pass'] += int((pred['donor_context'][mask] == target[mask]).sum())
            totals['no_context_one_pass'] += int((pred['no_context'][mask] == target[mask]).sum())
            totals['spatial'] += int((spatial[begin:begin + 8][mask] == target[mask]).sum())
            totals['frequency'] += int((target[mask] == frequent_token).sum())
    out.mkdir(parents=True, exist_ok=True)
    denominator = totals['true']
    result = {
        'checkpoint_sha256': sha256(args.checkpoint), 'data_identity': val['identity'],
        'split': 'validation', 'test_accessed': False,
        'seed': args.seed, 'per': args.per, 'images': len(ids),
        'frequent_train_token': frequent_token,
        'missing_tokens': denominator,
        'accuracy': {k: v / denominator for k, v in totals.items() if k != 'true'},
        'cross_entropy': {k: v / denominator for k, v in ce.items()},
        'correct_counts': {k: v for k, v in totals.items() if k != 'true'},
    }
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
