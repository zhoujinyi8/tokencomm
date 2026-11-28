"""Check the author's public MaskGIT checkpoint against the pinned source model."""
import argparse
import json
import sys
from pathlib import Path

import torch

from pretrained_vq import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('pretrained/official_tokcom/MaskGIT_ImageNet_256.pth'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from Network.transformer import MaskTransformer

    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(saved, dict) or 'model_state_dict' not in saved:
        raise ValueError(f"Expected model_state_dict; found {list(saved)[:20]}")
    state = saved['model_state_dict']
    model = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=1024,
                            depth=24, heads=16, mlp_dim=3072, dropout=0.1).eval()
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    extra = sorted(set(state) - set(target))
    wrong_shape = [key for key in target.keys() & state.keys()
                   if tuple(target[key].shape) != tuple(state[key].shape)]
    if missing or extra or wrong_shape:
        raise ValueError(json.dumps({'missing': missing[:20], 'extra': extra[:20],
                                     'wrong_shape': wrong_shape[:20]}))
    model.load_state_dict(state, strict=True, assign=True)
    result = {
        'scope': 'checkpoint/source structural match only; no 32-step generation or paper metric claim',
        'official_source_commit': '0422ed32dcde3347e0bb6697f343f7a9ff7b3214',
        'checkpoint_sha256': sha256(args.checkpoint),
        'outer_keys': sorted(saved.keys()), 'state_tensor_count': len(state),
        'model_parameter_count': sum(p.numel() for p in model.parameters()),
        'strict_state_load_passed': True,
        'token_embedding_shape': list(model.tok_emb.weight.shape),
        'positional_embedding_shape': list(model.pos_emb.shape),
        'first_parameter_name': next(iter(state)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
