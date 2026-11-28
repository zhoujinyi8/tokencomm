"""Development-only one-round packet retransmission priority diagnostic.

This is a receiver uncertainty heuristic, not the paper's full TCE or a
physical feedback channel. Retransmitted packets are assumed perfectly received.
The MaskGIT prediction is held fixed after retransmission, so this evaluates
direct packet repair only and does not claim iterative recovery benefits.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models import erase_packets
from pretrained_vq import sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path,
                   default=Path('data/imagenette_stage5/token_cache/val.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--per', type=float, default=.3)
    p.add_argument('--budget-packets', type=int, default=4)
    args = p.parse_args()
    if args.out.exists(): p.error('Output file already exists')
    if not 0 < args.per < 1 or not 0 < args.budget_packets <= 64:
        p.error('PER and packet budget out of range')
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    torch.set_num_threads(4)
    val = torch.load(args.data, map_location='cpu', weights_only=True)
    ids = val['ids']
    if ids.shape[1] != 256: raise ValueError('Expected 256-token cache')
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        order = torch.randperm(256)
        lost_packets = torch.rand(len(ids), 64) < args.per
        missing = torch.zeros_like(ids, dtype=torch.bool)
        missing[:, order] = lost_packets.repeat_interleave(4, dim=1)
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        if not torch.equal(missing, erase_packets(ids, args.per, 4)):
            raise ValueError('Reconstructed packet mask differs from shared erasure function')
    cfg = maskgit_class_cond_config.get_config()
    model = BidirectionalTransformer(
        num_image_tokens=257, num_codebook_vectors=2025,
        dim=cfg.transformer.num_embeds, n_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.intermediate_size,
        num_heads=cfg.transformer.num_heads,
        attention_dropout=cfg.transformer.dropout_rate,
        hidden_dropout=cfg.transformer.dropout_rate).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                     weights_only=True)['state_dict'], strict=True)
    rows = []
    with torch.inference_mode():
        for i, (target, mask) in enumerate(zip(ids, missing)):
            sequence = torch.cat((torch.tensor([2024]), target.masked_fill(mask, 2024)))[None]
            logits = model(sequence)[:, 1:, :1024][0]
            probs = logits.softmax(dim=-1)
            confidence, prediction = probs.max(dim=-1)
            wrong = (prediction != target) & mask
            packet_mask = mask[order].reshape(64, 4)
            if ((packet_mask.any(1)) != (packet_mask.all(1))).any():
                raise ValueError('Partial packet erasure')
            lost = packet_mask.any(1).nonzero().flatten().tolist()
            budget = min(args.budget_packets, len(lost))
            mean_conf = confidence[order].reshape(64, 4).mean(1)
            wrong_per_packet = wrong[order].reshape(64, 4).sum(1)
            selected = sorted(lost, key=lambda k: (float(mean_conf[k]), k))[:budget]
            corrected_uncertain = sum(int(wrong_per_packet[k]) for k in selected)
            wrong_total = int(wrong.sum())
            random_expected = budget / len(lost) * wrong_total if lost else 0.0
            oracle_corrected = sum(sorted((int(wrong_per_packet[k]) for k in lost),
                                          reverse=True)[:budget])
            rows.append({
                'source_index': i, 'class': val['class_names'][int(val['labels'][i])],
                'missing_tokens': int(mask.sum()), 'lost_packets': len(lost),
                'retransmitted_packets': budget, 'baseline_correct': int((~wrong & mask).sum()),
                'uncertain_packet_ids': selected,
                'uncertain_corrected_errors': corrected_uncertain,
                'random_expected_corrected_errors': random_expected,
                'oracle_corrected_errors': oracle_corrected,
            })
            if (i + 1) % 10 == 0: print(f'{i + 1}/{len(ids)}', flush=True)

    missing_total = sum(r['missing_tokens'] for r in rows)
    baseline = sum(r['baseline_correct'] for r in rows)
    uncertainty = sum(r['uncertain_corrected_errors'] for r in rows)
    random_expected = sum(r['random_expected_corrected_errors'] for r in rows)
    oracle = sum(r['oracle_corrected_errors'] for r in rows)
    rng = np.random.default_rng(20261028)
    samples = rng.integers(0, len(rows), size=(10000, len(rows)))
    paired_delta = np.array([r['uncertain_corrected_errors'] -
                             r['random_expected_corrected_errors'] for r in rows])
    denominators = np.array([r['missing_tokens'] for r in rows])
    draws = paired_delta[samples].sum(1) / denominators[samples].sum(1)
    result = {
        'scope': 'exploratory stage5 validation; direct perfect retransmission only; no test accessed',
        'seed': args.seed, 'per': args.per, 'packet_tokens': 4,
        'budget_packets_per_image': args.budget_packets, 'images': len(rows),
        'data_identity': val['identity'], 'checkpoint_sha256': sha256(args.checkpoint),
        'missing_tokens': missing_total, 'retransmitted_packets_total':
            sum(r['retransmitted_packets'] for r in rows),
        'initial_maskgit_accuracy': baseline / missing_total,
        'post_retransmit_accuracy': {
            'low_confidence': (baseline + uncertainty) / missing_total,
            'random_expectation': (baseline + random_expected) / missing_total,
            'oracle_privileged': (baseline + oracle) / missing_total,
        },
        'low_confidence_minus_random_expected': (uncertainty - random_expected) / missing_total,
        'image_bootstrap_95pct_difference': [float(v) for v in np.quantile(draws, [.025, .975])],
        'random_expectation_note': 'Exact expectation over uniform subsets of lost packets, not a single random draw.',
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['missing_tokens', 'initial_maskgit_accuracy',
          'post_retransmit_accuracy', 'low_confidence_minus_random_expected',
          'image_bootstrap_95pct_difference']}), flush=True)


if __name__ == '__main__': main()
