"""Check whether model confidence can complement spatial fill on held-out validation halves."""
import argparse
import json
from pathlib import Path

import torch

from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import sha256
from run_pretrained import spatial_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/stage7_context_maskgit_v2_full/best.pt')
    parser.add_argument('--data', default='data/imagenette_stage5/token_cache/val.pt')
    parser.add_argument('--out', default='runs/stage8_fusion_diagnostic')
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): parser.error('Use an empty output directory')
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = torch.load(args.data, map_location='cpu', weights_only=True)
    if ck['identity'] != data['identity']: raise ValueError('Dataset identity mismatch')
    ids, labels = data['ids'], data['labels']
    model = load_context_maskgit_v2(ck)
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, .3, 4)
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy())).long()
    best_score, best_id = [], []
    spatial_score = []
    with torch.inference_mode():
        for begin in range(0, len(ids), 8):
            target, mask = ids[begin:begin + 8], missing[begin:begin + 8]
            logits = model(target.masked_fill(mask, model.codes))
            score, pred = logits.max(-1)
            best_score.append(score)
            best_id.append(pred)
            spatial_score.append(logits.gather(-1, spatial[begin:begin + 8, :, None]).squeeze(-1))
    model_pred = torch.cat(best_id)
    gap = torch.cat(best_score) - torch.cat(spatial_score)
    order_within_class = torch.empty_like(labels)
    for label in labels.unique():
        indices = (labels == label).nonzero().flatten()
        order_within_class[indices] = torch.arange(len(indices))
    calibrate = order_within_class < 5
    check = ~calibrate
    thresholds = [0., .25, .5, 1., 2., 3., 4., 6., 8., 12.]

    def metrics(which, threshold):
        mask = missing & which[:, None]
        choose_model = (gap >= threshold) & mask
        pred = torch.where(choose_model, model_pred, spatial)
        count = int(mask.sum())
        return {
            'missing': count,
            'accuracy': int((pred[mask] == ids[mask]).sum()) / count,
            'model_fraction': int(choose_model.sum()) / count,
        }

    calibration = {str(threshold): metrics(calibrate, threshold) for threshold in thresholds}
    chosen = max(thresholds, key=lambda t: (calibration[str(t)]['accuracy'], t))
    comparison = {}
    for name, which in [('calibration', calibrate), ('validation_check', check)]:
        mask = missing & which[:, None]
        spatial_correct = (spatial[mask] == ids[mask])
        model_correct = (model_pred[mask] == ids[mask])
        comparison[name] = {
            'images': int(which.sum()), 'missing': int(mask.sum()),
            'spatial_accuracy': float(spatial_correct.float().mean()),
            'one_pass_accuracy': float(model_correct.float().mean()),
            'oracle_either_accuracy': float((spatial_correct | model_correct).float().mean()),
            'both_correct': int((spatial_correct & model_correct).sum()),
            'spatial_only_correct': int((spatial_correct & ~model_correct).sum()),
            'model_only_correct': int((~spatial_correct & model_correct).sum()),
            'selected_fusion': metrics(which, chosen),
        }
    result = {
        'split': 'stage5 validation only', 'test_accessed': False,
        'checkpoint_sha256': sha256(args.checkpoint), 'seed': args.seed,
        'thresholds': thresholds, 'calibration_by_threshold': calibration,
        'chosen_threshold': chosen, 'comparison': comparison,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
