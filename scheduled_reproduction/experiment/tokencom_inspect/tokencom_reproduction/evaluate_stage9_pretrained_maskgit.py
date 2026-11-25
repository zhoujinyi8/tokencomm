"""Development-only one-pass check of the official-weight MaskGIT PyTorch port.

Oracle label is a privileged upper-bound diagnostic, not a fair receiver setting.
Masked label uses no side information but is out of the pretrained model's training
distribution. Neither should be called a complete TokCom implementation.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import sha256
from run_pretrained import spatial_fill

IMAGENET_CLASS_IDS = {
    'tench': 0, 'English springer': 217, 'cassette player': 482,
    'chain saw': 491, 'church': 497, 'French horn': 566,
    'garbage truck': 569, 'gas pump': 571, 'golf ball': 574,
    'parachute': 701,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path,
                        default=Path('runs/stage7_context_maskgit_v2_full/best.pt'))
    parser.add_argument('--inferred-labels', type=Path,
                        help='Receiver-side predicted ImageNet classes from received tokens')
    parser.add_argument('--data', type=Path, default=Path('data/imagenette_stage5/token_cache/val.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--per', type=float, default=.3)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--all-images', action='store_true',
                        help='Use every stage5 validation image instead of first per class')
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output file already exists')
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    torch.set_num_threads(4)
    val = torch.load(args.data, map_location='cpu', weights_only=True)
    if val['class_names'] != list(IMAGENET_CLASS_IDS):
        raise ValueError('ImageNet class mapping differs from cache')
    # Selection is fixed by cache order, independent of model performance.
    if args.all_images:
        indices = list(range(len(val['ids'])))
    else:
        indices = []
        seen = set()
        for index, label in enumerate(val['labels'].tolist()):
            if label not in seen:
                indices.append(index)
                seen.add(label)
            if len(indices) >= args.limit: break
    ids = val['ids'][indices]
    labels = val['labels'][indices]
    inferred = None
    if args.inferred_labels:
        inferred = json.loads(args.inferred_labels.read_text(encoding='utf-8'))
        if inferred['seed'] != args.seed or inferred['per'] != args.per:
            raise ValueError('Inferred-label erasure protocol differs')
        if inferred['data_identity'] != val['identity']:
            raise ValueError('Inferred-label data identity differs')
        if [item['source_index'] for item in inferred['rows']] != indices:
            raise ValueError('Inferred-label sample selection differs')
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, args.per, 4)
    if not missing.any(): raise ValueError('No missing tokens')
    received = ids.masked_fill(missing, 2024)
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
    cfg = maskgit_class_cond_config.get_config()
    model = BidirectionalTransformer(
        num_image_tokens=257, num_codebook_vectors=2025,
        dim=cfg.transformer.num_embeds, n_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.intermediate_size,
        num_heads=cfg.transformer.num_heads,
        attention_dropout=cfg.transformer.dropout_rate,
        hidden_dropout=cfg.transformer.dropout_rate).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict']
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch: {incompatible}')
    del state
    baseline_saved = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=True)
    if baseline_saved['identity'] != val['identity']:
        raise ValueError('Baseline tokenizer/data identity differs')
    baseline = load_context_maskgit_v2(baseline_saved).eval()

    totals = {'oracle_label': 0, 'masked_label': 0, 'stage7_one_pass': 0, 'spatial': 0}
    ce_sums = {'oracle_label': 0., 'masked_label': 0., 'stage7_one_pass': 0.}
    if inferred:
        totals['receiver_inferred_label'] = 0
        ce_sums['receiver_inferred_label'] = 0.
    rows = []
    with torch.inference_mode():
        for row, (target, mask, data, local_label, source_index) in enumerate(
                zip(ids, missing, received, labels.tolist(), indices)):
            correct_class = IMAGENET_CLASS_IDS[val['class_names'][local_label]]
            outputs = {}
            conditions = [('oracle_label', 1024 + correct_class), ('masked_label', 2024)]
            if inferred:
                conditions.append(('receiver_inferred_label',
                                   1024 + inferred['rows'][row]['predicted_class']))
            for condition, prefix in conditions:
                sequence = torch.cat((torch.tensor([prefix]), data)).unsqueeze(0)
                logits = model(sequence)[:, 1:, :1024].squeeze(0)
                prediction = logits.argmax(-1)
                hits = int((prediction[mask] == target[mask]).sum())
                ce = float(F.cross_entropy(logits[mask], target[mask], reduction='sum'))
                totals[condition] += hits
                ce_sums[condition] += ce
                outputs[condition] = hits
            baseline_logits = baseline(target.masked_fill(mask, 1024).unsqueeze(0)).squeeze(0)
            baseline_hits = int((baseline_logits.argmax(-1)[mask] == target[mask]).sum())
            totals['stage7_one_pass'] += baseline_hits
            ce_sums['stage7_one_pass'] += float(F.cross_entropy(
                baseline_logits[mask], target[mask], reduction='sum'))
            outputs['stage7_one_pass'] = baseline_hits
            spatial_hits = int((spatial[row][mask] == target[mask]).sum())
            totals['spatial'] += spatial_hits
            rows.append({'source_index': source_index, 'class': val['class_names'][local_label],
                         'inferred_class': inferred['rows'][row]['predicted_class'] if inferred else None,
                         'missing_tokens': int(mask.sum()), 'correct': {**outputs, 'spatial': spatial_hits}})
            print(json.dumps(rows[-1]), flush=True)
    denominator = int(missing.sum())
    result = {
        'split': 'stage5 validation only', 'test_accessed': False,
        'scope': ('one-pass exploratory; oracle label is privileged, masked label is '
                  'out-of-distribution' + ('; inferred label uses extra receiver-side ResNet18'
                                           if inferred else '')),
        'checkpoint_sha256': sha256(args.checkpoint), 'data_identity': val['identity'],
        'baseline_checkpoint_sha256': sha256(args.baseline_checkpoint),
        'inferred_labels_sha256': sha256(args.inferred_labels) if inferred else None,
        'seed': args.seed, 'per': args.per,
        'selection': 'all validation images' if args.all_images else 'first validation image per class',
        'images': len(indices), 'missing_tokens': denominator,
        'correct_counts': totals, 'accuracy': {key: value / denominator for key, value in totals.items()},
        'cross_entropy': {key: value / denominator for key, value in ce_sums.items()},
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
