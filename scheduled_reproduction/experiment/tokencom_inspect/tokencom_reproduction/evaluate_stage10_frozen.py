"""Evaluate a frozen no-class-side-information MaskGIT receiver on new holdout.

Run only after protocol.json has been fixed from validation and the holdout token
cache has been created. Cases are independent and resumable by (PER, seed).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import sha256
from run_pretrained import spatial_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, help='Stop after this many new cases')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    required = {'version', 'holdout_manifest_sha256', 'tokenizer_sha256',
                'pretrained_checkpoint_sha256', 'stage7_checkpoint_sha256',
                'receiver', 'packet_tokens', 'pers', 'seeds', 'reference_commit'}
    if protocol.keys() != required or protocol['version'] != 1:
        raise ValueError('Unexpected frozen protocol schema')
    if protocol['receiver'] != 'one-pass masked class prefix 2024; argmax only erased tokens':
        raise ValueError('Receiver rule differs from frozen protocol')
    if protocol['packet_tokens'] != 4:
        raise ValueError('Packet size differs from frozen protocol')
    commit = subprocess.check_output(
        ['git', '-C', str(args.reference), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != protocol['reference_commit']:
        raise ValueError('Reference implementation commit differs')
    if sha256(args.checkpoint) != protocol['pretrained_checkpoint_sha256']:
        raise ValueError('Pretrained checkpoint hash mismatch')
    if sha256(args.baseline_checkpoint) != protocol['stage7_checkpoint_sha256']:
        raise ValueError('Baseline checkpoint hash mismatch')
    if sha256(args.data.parent.parent / 'manifest.json') != protocol['holdout_manifest_sha256']:
        raise ValueError('Holdout manifest hash mismatch')
    cache = torch.load(args.data, map_location='cpu', weights_only=True)
    if cache['identity']['manifest_sha256'] != protocol['holdout_manifest_sha256']:
        raise ValueError('Holdout token-cache manifest mismatch')
    if cache['identity']['tokenizer_sha256'] != protocol['tokenizer_sha256']:
        raise ValueError('Holdout tokenizer hash mismatch')
    ids = cache['ids']
    torch.set_num_threads(4)
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
    if baseline_saved['identity']['tokenizer_sha256'] != protocol['tokenizer_sha256']:
        raise ValueError('Baseline tokenizer mismatch')
    baseline = load_context_maskgit_v2(baseline_saved).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    saved_protocol = args.out / 'protocol.json'
    progress_path = args.out / 'progress.json'
    if args.resume:
        if not saved_protocol.exists() or not progress_path.exists():
            parser.error('No evaluation state to resume')
        if json.loads(saved_protocol.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Saved protocol differs')
        cases = json.loads(progress_path.read_text(encoding='utf-8'))['cases']
    else:
        if any(args.out.iterdir()): parser.error('Use an empty output directory')
        saved_protocol.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        cases = []
    completed = {(case['per'], case['seed']) for case in cases}
    new_cases = 0
    with torch.inference_mode():
        for per in protocol['pers']:
            for seed in protocol['seeds']:
                if (per, seed) in completed: continue
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    missing = erase_packets(ids, per, protocol['packet_tokens'])
                spatial = torch.from_numpy(spatial_fill(
                    ids.masked_fill(missing, 0).numpy(), missing.numpy()))
                totals = {'pretrained_masked_class': 0, 'stage7_one_pass': 0, 'spatial': 0}
                rows = []
                for index, (target, mask) in enumerate(zip(ids, missing)):
                    if not mask.any():
                        rows.append({'source_index': index, 'missing_tokens': 0,
                                     'correct': {key: 0 for key in totals}})
                        continue
                    received = target.masked_fill(mask, 2024)
                    sequence = torch.cat((torch.tensor([2024]), received)).unsqueeze(0)
                    prediction = model(sequence)[:, 1:, :1024].argmax(-1).squeeze(0)
                    completed_ids = target.clone()
                    completed_ids[mask] = prediction[mask]
                    if not torch.equal(completed_ids[~mask], target[~mask]):
                        raise RuntimeError('Known tokens changed')
                    old = baseline(target.masked_fill(mask, 1024).unsqueeze(0)).argmax(-1).squeeze(0)
                    correct = {
                        'pretrained_masked_class': int((prediction[mask] == target[mask]).sum()),
                        'stage7_one_pass': int((old[mask] == target[mask]).sum()),
                        'spatial': int((spatial[index][mask] == target[mask]).sum()),
                    }
                    for key, value in correct.items(): totals[key] += value
                    rows.append({'source_index': index, 'missing_tokens': int(mask.sum()),
                                 'correct': correct})
                denominator = int(missing.sum())
                case = {'per': per, 'seed': seed, 'images': len(ids),
                        'missing_tokens': denominator, 'correct_counts': totals,
                        'accuracy': {key: value / denominator for key, value in totals.items()},
                        'rows': rows}
                cases.append(case)
                temporary = progress_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps({'cases': cases}, indent=2), encoding='utf-8')
                temporary.replace(progress_path)
                print(json.dumps({key: value for key, value in case.items() if key != 'rows'}), flush=True)
                new_cases += 1
                if args.max_cases and new_cases >= args.max_cases: break
            if args.max_cases and new_cases >= args.max_cases: break
    finished = len(cases) == len(protocol['pers']) * len(protocol['seeds'])
    status = {'scope': 'new stage10 holdout; frozen protocol', 'test_accessed': True,
              'complete': finished, 'cases_completed': len(cases), 'cases_expected':
              len(protocol['pers']) * len(protocol['seeds'])}
    (args.out / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    if finished:
        (args.out / 'results.json').write_text(
            json.dumps({'protocol': protocol, 'cases': cases, **status}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
