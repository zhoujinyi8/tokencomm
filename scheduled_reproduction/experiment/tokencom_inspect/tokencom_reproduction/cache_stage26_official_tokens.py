"""Cache author's VQGAN token IDs for one stage5 dev image per class."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pretrained_vq import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Output exists')
    sys.path.insert(0, str(args.official_source.resolve()))
    from omegaconf import OmegaConf
    from Network.Taming.models.vqgan import VQModel

    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.official_source / 'pretrained_maskgit/VQGAN/model.yaml')
    model = VQModel(**cfg.model.params).eval()
    weight = Path('pretrained/official_tokcom/last.ckpt')
    incompatible = model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True)['state_dict'], strict=False)
    assert not incompatible.missing_keys
    data = Path('data/imagenette_stage5')
    cache = torch.load(data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
    first = {}
    for i, label in enumerate(cache['labels'].tolist()): first.setdefault(label, i)
    selected = [first[i] for i in range(10)]
    rows = []
    codes = []
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    with torch.inference_mode():
        for index in selected:
            record = cache['records'][index]
            path = data / record['image']
            if sha256(path) != record['sha256']: raise ValueError(f'Image changed: {path}')
            with Image.open(path) as source: image = preprocess(source.convert('RGB'))[None]
            _, _, (_, _, code) = model.encode(image)
            values = code.reshape(256).cpu().numpy().astype(np.int32)
            codes.append(values)
            rows.append({'source_index': index, 'label': int(cache['labels'][index]),
                         'class_name': cache['class_names'][int(cache['labels'][index])],
                         'image_sha256': record['sha256'], 'token_min': int(values.min()),
                         'token_max': int(values.max())})
    result = {'scope': '10 stage5 dev images, author VQGAN, demo Resize((256,256))+ToTensor',
              'selected_indices': selected, 'vqgan_sha256': sha256(weight), 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=np.stack(codes), metadata=json.dumps(result))
    print(json.dumps({'images': len(rows), 'output': str(args.out), 'classes': [r['class_name'] for r in rows]}))


if __name__ == '__main__': main()
