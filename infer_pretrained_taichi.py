"""Single-image inference for the frozen-ResNet18 TaiChi bridge."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights
from cache_pretrained_vision import load_backbone,features
from pretrained_taichi import load_feature_taichi
from run import encode,decode


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--checkpoint',default='runs/stage3_pretrained_taichi_v2/best.pt')
    p.add_argument('--question',default='What is the main object in this image?')
    p.add_argument('--channel',choices=['awgn','rayleigh'],default='awgn');p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--out',default='runs/stage3_single_image');p.add_argument('--seed',type=int,default=7302)
    a=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=load_feature_taichi(ck).eval();backbone,digest=load_backbone()
    if digest!=ck['feature_identity']['weight_sha256']:raise ValueError('Feature backbone differs from training')
    with Image.open(a.image) as raw:
        im=raw.convert('RGB');x=ResNet18_Weights.IMAGENET1K_V1.transforms()(im).unsqueeze(0)
    q=torch.tensor([encode(a.question)]);torch.manual_seed(a.seed)
    answer=decode(model.generate(features(backbone,x),q,a.snr,a.channel,max_new_tokens=24)[0].tolist())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);im.save(out/'input.png')
    result={'image':a.image,'question':a.question,'answer':answer,'channel':a.channel,'snr_db':a.snr,
            'checkpoint':a.checkpoint,'selected_step':ck['step'],'feature_weight_sha256':digest,
            'scope':'10-class QA bridge, not a general-purpose VLM','seed':a.seed}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
