"""Development-set ResNet18 feature diagnostic; not paper CLIP/LPIPS metric."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights

from cache_pretrained_vision import load_backbone
from cache_real_tokens import load_image
from models import erase_packets
from pretrained_vq import PretrainedVQGAN, sha256
from run_pretrained import spatial_fill


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/imagenette_stage5'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--images', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261021)
    args = p.parse_args()
    if args.out.exists(): p.error('Output already exists')
    if not 1 <= args.images <= 100: p.error('images must be 1..100')
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    torch.set_num_threads(4)
    val = torch.load(args.data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    ids = val['ids']
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        missing = erase_packets(ids, .3, 4)
    reference_rows = json.loads((Path('runs/stage11_pretrained_reconstruction_full/results.json'))
                                .read_text(encoding='utf-8'))['rows']
    assert len(reference_rows) == 100
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
    tokenizer = PretrainedVQGAN().eval()
    backbone, backbone_sha256 = load_backbone()
    feature_model = torch.nn.Sequential(*list(backbone.children())[:-1]).eval()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rows = []
    progress = args.out.with_suffix('.progress.jsonl')
    if progress.exists():
        rows = [json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines() if line]
        if [r['source_index'] for r in rows] != list(range(len(rows))):
            raise ValueError('Progress is not a prefix of selected images')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(len(rows), args.images):
            target = ids[index]
            mask = missing[index]
            received = target.masked_fill(mask, 2024)
            logits = model(torch.cat((torch.tensor([2024]), received))[None])[:, 1:, :1024]
            pretrained_ids = torch.where(mask, logits.argmax(-1)[0], target)
            spatial_ids = torch.from_numpy(spatial_fill(
                target.masked_fill(mask, 0)[None].numpy(), mask[None].numpy())[0]).long()
            record = val['records'][index]
            path = args.data / record['image']
            if sha256(path) != record['sha256']:
                raise ValueError(f'Image changed: {path}')
            original = load_image(path)[None]
            decoded = tokenizer.decode(torch.stack((target, pretrained_ids, spatial_ids)))
            images = torch.cat((original, decoded))
            feature = feature_model(transform(images)).flatten(1)
            cos = F.cosine_similarity(feature[0:1], feature[1:], dim=1).tolist()
            classes = backbone(transform(images)).argmax(-1).tolist()
            ref = reference_rows[index]
            assert ref['source_index'] == index and ref['image_sha256'] == record['sha256']
            assert ref['missing_tokens'] == int(mask.sum())
            assert ref['token_hits']['pretrained_masked_class'] == int((pretrained_ids[mask] == target[mask]).sum())
            assert ref['token_hits']['spatial'] == int((spatial_ids[mask] == target[mask]).sum())
            row = {
                'source_index': index, 'image_sha256': record['sha256'],
                'missing_tokens': int(mask.sum()),
                'feature_cosine_to_original': dict(zip(
                    ('clean_vqgan', 'pretrained_masked_class', 'spatial'), cos)),
                'resnet_top1_class': dict(zip(
                    ('original', 'clean_vqgan', 'pretrained_masked_class', 'spatial'), classes)),
                'top1_agreement_to_original': dict(zip(
                    ('clean_vqgan', 'pretrained_masked_class', 'spatial'),
                    [classes[0] == x for x in classes[1:]]))}
            rows.append(row)
            with progress.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps({'image': index, 'cosine': row['feature_cosine_to_original']}), flush=True)
    methods = list(rows[0]['feature_cosine_to_original'])
    result = {
        'scope': 'development-only frozen ResNet18 feature proxy; not CLIP/LPIPS',
        'images': len(rows), 'seed': args.seed, 'per': .3,
        'data_identity': val['identity'],
        'maskgit_checkpoint_sha256': sha256(args.checkpoint),
        'resnet18_sha256': backbone_sha256,
        'mean_feature_cosine_to_original': {
            name: sum(r['feature_cosine_to_original'][name] for r in rows) / len(rows)
            for name in methods},
        'top1_agreement_to_original': {
            name: sum(r['top1_agreement_to_original'][name] for r in rows) / len(rows)
            for name in methods},
        'rows': rows,
        'caution': 'ResNet18 pretraining used ImageNet; Imagenette is an ImageNet subset. '
                   'Feature similarity is a proxy, not proof of semantic preservation.'}
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'means': result['mean_feature_cosine_to_original'],
                      'top1_agreement': result['top1_agreement_to_original']}), flush=True)


if __name__ == '__main__': main()
