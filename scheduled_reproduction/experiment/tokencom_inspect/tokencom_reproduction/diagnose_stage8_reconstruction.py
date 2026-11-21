"""Compare VQGAN reconstruction PSNR on one preselected image per class (validation only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cache_real_tokens import load_image
from context_maskgit_v2 import load_context_maskgit_v2
from models import erase_packets
from pretrained_vq import PretrainedVQGAN, sha256
from run_pretrained import spatial_fill


def psnr(reference, output):
    mse = float((reference - output).square().mean())
    return float(-10. * np.log10(max(mse, 1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/stage7_context_maskgit_v2_full/best.pt')
    parser.add_argument('--data', default='data/imagenette_stage5')
    parser.add_argument('--out', default='runs/stage8_reconstruction_diagnostic')
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): parser.error('Use an empty output directory')
    torch.set_num_threads(4)
    root = Path(args.data)
    cache = torch.load(root / 'token_cache' / 'val.pt', map_location='cpu', weights_only=True)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ck['identity'] != cache['identity']: raise ValueError('Token cache differs from training identity')
    labels = cache['labels']
    chosen = [int((labels == label).nonzero().flatten()[0]) for label in range(len(cache['class_names']))]
    ids = cache['ids'][chosen]
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, .3, 4)
    model = load_context_maskgit_v2(ck).eval()
    receiver = ids.masked_fill(missing, model.codes)
    tokenizer = PretrainedVQGAN().eval()
    spatial = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy())).long()
    results = []
    with torch.inference_mode():
        one_pass = torch.where(missing, model(receiver).argmax(-1), ids)
        two_steps = model.recover(receiver, missing, steps=2)
        four_steps = model.recover(receiver, missing, steps=4)
        iterative = model.recover(receiver, missing, steps=8)
        for row, position in enumerate(chosen):
            record = cache['records'][position]
            raw_path = root / record['image']
            if sha256(raw_path) != record['sha256']:
                raise ValueError('Validation image changed')
            image = load_image(raw_path)[None]
            scores = {}
            for name, tokens in [
                ('clean_vqgan', ids[row]), ('spatial', spatial[row]),
                ('one_pass', one_pass[row]), ('two_steps', two_steps[row]),
                ('four_steps', four_steps[row]), ('eight_steps', iterative[row]),
            ]:
                output = tokenizer.decode(tokens[None])
                scores[name] = psnr(image, output)
            results.append({
                'class': cache['class_names'][int(labels[position])],
                'image_sha256': record['sha256'],
                'erased_tokens': int(missing[row].sum()),
                'psnr_db': scores,
            })
            print(json.dumps(results[-1]), flush=True)
    means = {key: sum(row['psnr_db'][key] for row in results) / len(results)
             for key in results[0]['psnr_db']}
    summary = {
        'split': 'stage5 validation', 'test_accessed': False,
        'selection': 'first validation image of each of ten classes',
        'checkpoint_sha256': sha256(args.checkpoint),
        'seed': args.seed, 'per': .3, 'rows': results, 'mean_psnr_db': means,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'mean_psnr_db': means}), flush=True)


if __name__ == '__main__':
    main()
