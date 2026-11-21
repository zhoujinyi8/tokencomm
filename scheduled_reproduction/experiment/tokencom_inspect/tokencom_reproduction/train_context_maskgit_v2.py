"""Train the no-CMI 2-D MaskGIT receiver using train/validation caches only.

This entry intentionally never reads ``test.pt``.  Pick the model checkpoint by
validation cross-entropy, then run a separately reviewed final evaluation on a
previously unopened holdout split.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v2 import ContextMaskGITv2, load_context_maskgit_v2
from models import erase_packets
from run_pretrained import spatial_fill


@torch.inference_mode()
def evaluate(model, ids, seed, per, batch_size):
    """Use one fixed packet-erasure realization for reproducible validation."""
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        missing = erase_packets(ids, per, 4)
    total_loss = 0.0
    recovered = []
    for start in range(0, len(ids), batch_size):
        target = ids[start:start + batch_size]
        mask = missing[start:start + batch_size]
        receiver = target.masked_fill(mask, model.codes)
        logits = model(receiver)
        total_loss += float(F.cross_entropy(logits[mask], target[mask], reduction='sum'))
        prediction = model.recover(receiver, mask)
        if not torch.equal(prediction[~mask], target[~mask]):
            raise RuntimeError('Receiver modified a token that was not erased')
        recovered.append(prediction)
    prediction = torch.cat(recovered)
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
    return {
        'seed': seed, 'per': per, 'cross_entropy': total_loss / int(missing.sum()),
        'missing_tokens': int(missing.sum()),
        'accuracy': float((prediction[missing] == ids[missing]).float().mean()),
        'spatial_accuracy': float((spatial[missing] == ids[missing]).float().mean()),
    }


def checkpoint(model, optimizer, scheduler, step, run_config, identity, best_val_ce, best_step):
    return {
        'model': 'context_maskgit_v2', 'config': model.config, 'state': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'rng_cpu': torch.get_rng_state(), 'step': step, 'run_config': run_config,
        'identity': identity, 'best_val_ce': best_val_ce, 'best_step': best_step,
        'scope': 'no-CMI receiver; train/validation only; no final-test access',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/imagenette_stage5/token_cache')
    p.add_argument('--out', default='runs/stage7_context_maskgit_v2')
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--per', type=float, default=.3)
    p.add_argument('--eval-every', type=int, default=400)
    p.add_argument('--d', type=int, default=192)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--local-blocks', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if not 0 < a.per <= 1: p.error('--per must be in (0, 1]')
    if a.steps < 1 or a.eval_every < 1: p.error('--steps and --eval-every must be positive')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    # Deliberately do not construct or probe root / 'test.pt'.
    train = torch.load(root / 'train.pt', map_location='cpu', weights_only=True)
    val = torch.load(root / 'val.pt', map_location='cpu', weights_only=True)
    if train['identity'] != val['identity']:
        raise ValueError('Training and validation token caches have different identities')
    identity = train['identity']
    model = ContextMaskGITv2(a.d, a.layers, a.local_blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.steps, eta_min=a.lr / 10)
    config = {
        'args': vars(a), 'train_images': len(train['ids']), 'validation_images': len(val['ids']),
        'tokenizer_identity': identity, 'condition': 'none', 'packet_tokens': 4,
        'selection_metric': 'validation cross entropy at fixed seed 20261020',
        'test_accessed': False,
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    initial = evaluate(model, val['ids'], 20261020, a.per, a.batch_size)
    (out / 'before.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    best_ce, best_step = initial['cross_entropy'], 0
    torch.save(checkpoint(model, optimizer, scheduler, 0, config, identity, best_ce, best_step), out / 'best.pt')
    started = time.monotonic()
    with (out / 'history.jsonl').open('w', encoding='utf-8', buffering=1) as history:
        for step in range(1, a.steps + 1):
            model.train()
            index = torch.randint(len(train['ids']), (a.batch_size,))
            target = train['ids'][index]
            loss, mask = model.loss(target, packet_tokens=4, per=a.per)
            if not mask.any() or not torch.isfinite(loss): raise RuntimeError('Invalid masked-token training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            row = {'step': step, 'loss': float(loss.detach()), 'lr': scheduler.get_last_lr()[0],
                   'elapsed_seconds': time.monotonic() - started}
            if step % a.eval_every == 0 or step == a.steps:
                row['validation'] = evaluate(model, val['ids'], 20261020, a.per, a.batch_size)
                if row['validation']['cross_entropy'] < best_ce:
                    best_ce, best_step = row['validation']['cross_entropy'], step
                    torch.save(checkpoint(model, optimizer, scheduler, step, config, identity, best_ce, best_step), out / 'best.pt')
                torch.save(checkpoint(model, optimizer, scheduler, step, config, identity, best_ce, best_step), out / 'last.pt')
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + '\n')
    selected = load_context_maskgit_v2(torch.load(out / 'best.pt', map_location='cpu', weights_only=True))
    result = {'completed': True, 'best_step': best_step, 'best_validation': evaluate(selected, val['ids'], 20261021, a.per, a.batch_size),
              'test_accessed': False}
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'status.json').write_text(json.dumps({'completed': True, 'best_step': best_step, 'test_accessed': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
