"""Exploratory image reconstruction check on stage5 validation only.

Uses the same seed, packet masks, and one-pass masked-class MaskGIT rule as
stage10 validation. Never reads a test split. PSNR is a pixel metric, not a
semantic fidelity metric.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

from cache_real_tokens import load_image
from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import PretrainedVQGAN, sha256
from run_pretrained import spatial_fill


def psnr(reference: torch.Tensor, output: torch.Tensor) -> float:
    mse = float((reference - output).square().mean())
    return -10.0 * math.log10(max(mse, 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline-checkpoint', type=Path,
                   default=Path('runs/stage7_context_maskgit_v2_full/best.pt'))
    p.add_argument('--data', type=Path, default=Path('data/imagenette_stage5'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--per', type=float, default=.3)
    p.add_argument('--all-images', action='store_true',
                   help='Evaluate all 100 stage5 validation images')
    args = p.parse_args()
    if args.out.exists():
        p.error('Output file already exists')
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    torch.set_num_threads(4)
    val = torch.load(args.data / 'token_cache' / 'val.pt', map_location='cpu', weights_only=True)
    labels = val['labels']
    selected = (list(range(len(labels))) if args.all_images else
                [int((labels == label).nonzero().flatten()[0])
                 for label in range(len(val['class_names']))])
    ids = val['ids']
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, args.per, 4)
    protocol = {
        'data_identity': val['identity'], 'selected_indices': selected,
        'seed': args.seed, 'per': args.per, 'packet_tokens': 4,
        'checkpoint_sha256': sha256(args.checkpoint),
        'baseline_checkpoint_sha256': sha256(args.baseline_checkpoint),
    }
    progress_path = args.out.with_suffix('.progress.jsonl')
    protocol_path = args.out.with_suffix('.protocol.json')
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Saved protocol differs from current invocation')
    else:
        if progress_path.exists(): raise ValueError('Progress without protocol')
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = ([json.loads(line) for line in progress_path.read_text(encoding='utf-8').splitlines()
             if line.strip()] if progress_path.exists() else [])
    if [r['source_index'] for r in rows] != selected[:len(rows)]:
        raise ValueError('Saved progress is not a prefix of selected images')
    print(f'resuming {len(rows)}/{len(selected)} images', flush=True)

    cfg = maskgit_class_cond_config.get_config()
    model = BidirectionalTransformer(
        num_image_tokens=257, num_codebook_vectors=2025,
        dim=cfg.transformer.num_embeds, n_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.intermediate_size,
        num_heads=cfg.transformer.num_heads,
        attention_dropout=cfg.transformer.dropout_rate,
        hidden_dropout=cfg.transformer.dropout_rate).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict(state, strict=True)
    del state
    baseline_saved = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=True)
    if baseline_saved['identity'] != val['identity']:
        raise ValueError('Baseline/tokenizer data identity differs')
    baseline = load_context_maskgit_v2(baseline_saved).eval()
    tokenizer = PretrainedVQGAN().eval()
    with torch.inference_mode():
        for index in selected[len(rows):]:
            target = ids[index]
            mask = missing[index]
            received = target.masked_fill(mask, 2024)
            sequence = torch.cat((torch.tensor([2024]), received)).unsqueeze(0)
            prediction = model(sequence)[:, 1:, :1024].argmax(-1)[0]
            pretrained_ids = torch.where(mask, prediction, target)
            small_prediction = baseline(target.masked_fill(mask, 1024)[None]).argmax(-1)[0]
            small_ids = torch.where(mask, small_prediction, target)
            spatial_ids = torch.from_numpy(spatial_fill(
                target.masked_fill(mask, 0)[None].numpy(), mask[None].numpy())[0]).long()

            record = val['records'][index]
            path = args.data / record['image']
            if sha256(path) != record['sha256']:
                raise ValueError(f'Validation image changed: {path}')
            original = load_image(path)[None]
            names = ('clean_vqgan', 'pretrained_masked_class', 'stage7_one_pass', 'spatial')
            codes = torch.stack((target, pretrained_ids, small_ids, spatial_ids))
            decoded_batch = tokenizer.decode(codes)
            decoded = {name: decoded_batch[i:i + 1] for i, name in enumerate(names)}
            clean = decoded['clean_vqgan']
            rows.append({
                'source_index': index, 'class': val['class_names'][int(labels[index])],
                'image_sha256': record['sha256'], 'missing_tokens': int(mask.sum()),
                'token_hits': {
                    'pretrained_masked_class': int((pretrained_ids[mask] == target[mask]).sum()),
                    'stage7_one_pass': int((small_ids[mask] == target[mask]).sum()),
                    'spatial': int((spatial_ids[mask] == target[mask]).sum())},
                'psnr_to_original_db': {name: psnr(original, image) for name, image in decoded.items()},
                'psnr_to_clean_vqgan_db': {
                    name: psnr(clean, image) for name, image in decoded.items()
                    if name != 'clean_vqgan'},
            })
            with progress_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(rows[-1]) + '\n')
            print(json.dumps(rows[-1]), flush=True)

    names = list(rows[0]['psnr_to_original_db'])
    result = {
        'scope': 'exploratory stage5 validation reconstruction; no test accessed',
        'seed': args.seed, 'per': args.per, 'packet_tokens': 4,
        'selection': ('all cached validation images' if args.all_images else
                      'first cached validation image per class; mask generated on all 100 images'),
        'images': len(rows), 'data_identity': val['identity'],
        'checkpoint_sha256': sha256(args.checkpoint),
        'baseline_checkpoint_sha256': sha256(args.baseline_checkpoint),
        'mean_psnr_to_original_db': {
            name: sum(r['psnr_to_original_db'][name] for r in rows) / len(rows) for name in names},
        'mean_psnr_to_clean_vqgan_db': {
            name: sum(r['psnr_to_clean_vqgan_db'][name] for r in rows) / len(rows)
            for name in rows[0]['psnr_to_clean_vqgan_db']},
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'means': result['mean_psnr_to_original_db']}), flush=True)


if __name__ == '__main__':
    main()
