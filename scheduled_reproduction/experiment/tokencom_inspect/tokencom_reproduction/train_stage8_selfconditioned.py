"""Short diagnostic fine-tuning with model-filled tokens in the training context."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import sha256
from train_context_maskgit_v2 import evaluate


def self_conditioned_loss(model, ids, per):
    missing = erase_packets(ids, per, 4)
    received = ids.masked_fill(missing, model.codes)
    initial_logits = model(received)
    initial_loss = F.cross_entropy(initial_logits[missing], ids[missing])
    with torch.no_grad():
        confidence, prediction = initial_logits.softmax(-1).max(-1)
        partially_filled = received.clone()
        remaining = missing.clone()
        for row in range(len(ids)):
            candidates = missing[row].nonzero().flatten()
            fraction = float(torch.empty(()).uniform_(.25, .75))
            take = min(len(candidates) - 1, int(len(candidates) * fraction))
            if take:
                chosen = candidates[confidence[row, candidates].topk(take).indices]
                partially_filled[row, chosen] = prediction[row, chosen]
                remaining[row, chosen] = False
    refined_logits = model(partially_filled)
    refined_loss = F.cross_entropy(refined_logits[remaining], ids[remaining])
    return .5 * initial_loss + .5 * refined_loss, float(initial_loss.detach()), float(refined_loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', default='runs/stage7_context_maskgit_v2_full/best.pt')
    parser.add_argument('--data', default='data/imagenette_stage5/token_cache')
    parser.add_argument('--out', default='runs/stage8_selfconditioned')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--run-steps', type=int, default=None,
                        help='Maximum additional optimizer steps in this invocation')
    parser.add_argument('--resume', action='store_true', help='Resume from out/last.pt')
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1: parser.error('Steps must be positive')
    if args.run_steps is not None and args.run_steps < 1: parser.error('--run-steps must be positive')
    out = Path(args.out)
    if not args.resume and out.exists() and any(out.iterdir()): parser.error('Use an empty output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    init = torch.load(args.init, map_location='cpu', weights_only=True)
    root = Path(args.data)
    train = torch.load(root / 'train.pt', map_location='cpu', weights_only=True)
    val = torch.load(root / 'val.pt', map_location='cpu', weights_only=True)
    if init['identity'] != train['identity'] or train['identity'] != val['identity']:
        raise ValueError('Tokenizer or data identity mismatch')
    model = load_context_maskgit_v2(init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 10)
    run_args = {key: value for key, value in vars(args).items() if key not in ('run_steps', 'resume')}
    config = {
        'args': run_args, 'initial_checkpoint_sha256': sha256(args.init),
        'training_images': len(train['ids']), 'validation_images': len(val['ids']),
        'identity': train['identity'], 'test_accessed': False,
        'objective': 'equal initial and self-conditioned masked cross entropy',
        'selection': 'highest fixed-seed validation eight-step exact recovery; CE tie-breaker',
    }
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if args.resume:
        if not (out / 'last.pt').exists(): parser.error('No last.pt checkpoint to resume')
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration differs from the saved run')
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

    def validate():
        return evaluate(model, val['ids'], 20261020, .3, args.batch_size)

    if args.resume:
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng_cpu'])
        start_step = saved['step']
        baseline = json.loads((out / 'before.json').read_text(encoding='utf-8'))
        best = tuple(saved['best_score'])
        best_step = saved['best_step']
    else:
        baseline = validate()
        (out / 'before.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
        best = (baseline['accuracy'], -baseline['cross_entropy'])
        best_step = 0
        start_step = 0

    def checkpoint(step):
        return {
            'model': 'context_maskgit_v2', 'config': model.config,
            'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'rng_cpu': torch.get_rng_state(),
            'step': step, 'best_step': best_step, 'best_score': list(best),
            'identity': train['identity'],
            'run_config': config, 'scope': 'self-conditioned development experiment',
        }

    if not args.resume:
        torch.save(checkpoint(0), out / 'best.pt')
    started = time.monotonic()
    target_step = min(args.steps, start_step + (args.run_steps or args.steps))
    with (out / 'history.jsonl').open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as history:
        for step in range(start_step + 1, target_step + 1):
            model.train()
            sample = torch.randint(len(train['ids']), (args.batch_size,))
            target = train['ids'][sample]
            loss, first, refined = self_conditioned_loss(model, target, .3)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
            row = {'step': step, 'loss': float(loss.detach()), 'first_loss': first,
                   'refined_loss': refined, 'elapsed_seconds': time.monotonic() - started}
            if step % args.eval_every == 0 or step == target_step:
                row['validation'] = validate()
                score = (row['validation']['accuracy'], -row['validation']['cross_entropy'])
                if score > best:
                    best = score
                    best_step = step
                    torch.save(checkpoint(step), out / 'best.pt')
                torch.save(checkpoint(step), out / 'last.pt')
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + '\n')
    final = {'completed': target_step == args.steps, 'current_step': target_step,
             'best_step': best_step, 'before': baseline,
             'last_validation': validate(), 'test_accessed': False}
    (out / ('results.json' if final['completed'] else 'status.json')).write_text(
        json.dumps(final, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
