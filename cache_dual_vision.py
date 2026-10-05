"""Cache frozen public ViT-B/16 and ConvNeXt-Tiny features without opening the test split."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.models import (vit_b_16, ViT_B_16_Weights,
                                convnext_tiny, ConvNeXt_Tiny_Weights)
from PIL import Image

from pretrained_vq import sha256


def download_and_load(factory, weights, filename, prefix):
    root = Path(__file__).resolve().parent
    path = root / 'pretrained' / filename
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        torch.hub.download_url_to_file(weights.url, str(path), hash_prefix=prefix, progress=True)
    digest = sha256(path)
    if not digest.startswith(prefix):
        raise ValueError(f'Public weight hash mismatch for {filename}')
    model = factory(weights=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval().requires_grad_(False), digest


def load_backbones():
    vit, vit_hash = download_and_load(vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1,
                                      'vit_b_16-c867db91.pth', 'c867db91')
    conv, conv_hash = download_and_load(convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                                        'convnext_tiny-983f1562.pth', '983f1562')
    return vit, conv, vit_hash, conv_hash


@torch.inference_mode()
def extract(vit, conv, x):
    v = vit._process_input(x)
    cls = vit.class_token.expand(x.shape[0], -1, -1)
    v = vit.encoder(torch.cat([cls, v], dim=1))[:, 1:]
    v = v.transpose(1, 2).reshape(x.shape[0], 768, 14, 14)
    v = F.adaptive_avg_pool2d(v, (4, 4))
    c = conv.features(x)
    if c.shape[1:] != (768, 7, 7):
        raise ValueError(f'Unexpected ConvNeXt feature shape: {tuple(c.shape)}')
    return v, c


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/imagenette_stage7')
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if 'test' in a.splits:
        p.error('Stage-8 development cache must not open test; use a separate frozen evaluator later')
    torch.set_num_threads(4)
    root = Path(a.data); out = root / 'dual_vision_features'; out.mkdir(exist_ok=True)
    vit, conv, vh, ch = load_backbones()
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    identity = {
        'vit_backbone': 'torchvision ViT-B/16 IMAGENET1K_V1', 'vit_weight_sha256': vh,
        'conv_backbone': 'torchvision ConvNeXt-Tiny IMAGENET1K_V1', 'conv_weight_sha256': ch,
        'manifest_sha256': sha256(root / 'manifest.json'), 'preprocessing': str(transform),
        'vit_feature': 'encoder patch tokens without CLS, reshape14x14, adaptive pool4x4',
        'conv_feature': 'final features, 768x7x7', 'frozen': True, 'shared_backbone': False,
        'test_opened': False,
        'pretraining_overlap': 'ImageNet pretrained; Imagenette is an ImageNet subset; no unseen-pretraining claim'
    }
    for split in a.splits:
        dst = out / f'{split}.pt'
        if dst.exists():
            old = torch.load(dst, weights_only=True)
            if old['identity'] != identity: raise ValueError('Existing cache identity differs')
            print('Verified existing', split, flush=True); continue
        records = [json.loads(s) for s in (root / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        vs, cs = [], []
        for start in range(0, len(records), a.batch_size):
            ims = []
            for r in records[start:start+a.batch_size]:
                with Image.open(root / r['image']) as im: ims.append(transform(im.convert('RGB')))
            v, c = extract(vit, conv, torch.stack(ims)); vs.append(v.to(torch.float16)); cs.append(c.to(torch.float16))
            if start % 40 == 0: print(split, start, '/', len(records), flush=True)
        torch.save({'vit': torch.cat(vs), 'conv': torch.cat(cs),
                    'labels': torch.tensor([r['label'] for r in records]), 'records': records,
                    'identity': identity}, dst)
    (out / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps({'splits': a.splits, 'test_opened': False,
                                                   'images': sum(len(torch.load(out/f'{s}.pt', weights_only=True)['labels']) for s in a.splits)}, indent=2), encoding='utf-8')
    print('Saved distinct frozen dual-backbone features; test opened: false', flush=True)


if __name__ == '__main__': main()
