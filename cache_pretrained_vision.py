"""Frozen public ResNet18 dual-resolution feature bridge, not paper CLIP backbones."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from pretrained_vq import sha256


def load_backbone():
    root=Path(__file__).resolve().parent
    path=root/'pretrained/resnet18-f37072fd.pth'
    if not path.exists():
        torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url,str(path),hash_prefix='f37072fd',progress=False)
    digest=sha256(path)
    if not digest.startswith('f37072fd'):raise ValueError('Public weight hash mismatch')
    model=resnet18(weights=None);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False)
    return model,digest


def features(model,x):
    def trunk(t):return model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(t)))))))
    hr=trunk(x)
    lr=model.layer4(trunk(F.interpolate(x,(112,112),mode='bilinear',align_corners=False)))
    return F.adaptive_avg_pool2d(lr,(4,4)),F.adaptive_avg_pool2d(hr,(8,8))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/imagenette_pilot');p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    a=p.parse_args()
    torch.set_num_threads(4)
    root=Path(a.data); out=root/'resnet18_features';out.mkdir(exist_ok=True)
    model,digest=load_backbone(); transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    identity={'backbone':'torchvision ResNet18 IMAGENET1K_V1','weight_sha256':digest,'manifest_sha256':sha256(root/'manifest.json'),
              'preprocessing':str(transform),'lr':'normalized 224 crop resized to112, layer4, pool4x4',
              'hr':'normalized224 crop, layer3, pool8x8','frozen':True,'shared_backbone':True,
              'pretraining_overlap':'ImageNet pretrained; Imagenette is ImageNet subset; no unseen-pretraining claim'}
    for split in a.splits:
        dst=out/(split+'.pt')
        if dst.exists():
            old=torch.load(dst,weights_only=True)
            if old['identity']!=identity:raise ValueError('Existing cache identity differs')
            print('Verified existing',split,flush=True);continue
        records=[json.loads(s) for s in (root/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        lo=[];hi=[]
        for start in range(0,len(records),8):
            images=[]
            for r in records[start:start+8]:
                with Image.open(root/r['image']) as im:images.append(transform(im.convert('RGB')))
            lr,hr=features(model,torch.stack(images));lo.append(lr);hi.append(hr)
            if start%64==0:print(split,start,'/',len(records),flush=True)
        torch.save({'lr':torch.cat(lo),'hr':torch.cat(hi),'labels':torch.tensor([r['label'] for r in records]),
                    'records':records,'identity':identity},dst)
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print('Saved frozen features. Splits:',a.splits,'test opened:', 'test' in a.splits,flush=True)


if __name__=='__main__':main()
