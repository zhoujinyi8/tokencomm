"""Open the held-out split only after the committed Stage-10 protocol is frozen."""
import json
from pathlib import Path
import torch
from torchvision.models import ViT_B_16_Weights
from PIL import Image

from cache_dual_vision import load_backbones, extract
from pretrained_vq import sha256


@torch.inference_mode()
def main():
    root=Path(__file__).resolve().parent; protocol_path=root/'STAGE10_FINAL_TEST_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    checkpoint=root/protocol['checkpoint']
    if sha256(checkpoint).upper()!=protocol['checkpoint_sha256']: raise ValueError('Frozen checkpoint hash mismatch')
    source=root/'data/imagenette_stage7'; records=[json.loads(x) for x in (source/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=protocol['expected_test_images']: raise ValueError('Unexpected held-out size')
    counts=torch.bincount(torch.tensor([r['label'] for r in records]),minlength=10)
    if not torch.equal(counts,torch.full((10,),15)): raise ValueError(f'Unexpected class balance: {counts.tolist()}')
    out=root/'data/stage10_final_test'; out.mkdir(exist_ok=True); dst=out/'test.pt'
    if dst.exists(): raise FileExistsError('Final test cache already exists; refusing to overwrite')
    torch.set_num_threads(4); vit,conv,vh,ch=load_backbones(); transform=ViT_B_16_Weights.IMAGENET1K_V1.transforms(); vs=[]; cs=[]
    for start in range(0,len(records),8):
        images=[]
        for row in records[start:start+8]:
            with Image.open(source/row['image']) as im: images.append(transform(im.convert('RGB')))
        v,c=extract(vit,conv,torch.stack(images)); vs.append(v.half()); cs.append(c.half()); print(start,'/',len(records),flush=True)
    identity={'purpose':'Stage-10 preregistered one-time final test','protocol_sha256':sha256(protocol_path),
              'checkpoint_sha256':sha256(checkpoint),'test_manifest_sha256':sha256(source/'test.jsonl'),
              'vit_weight_sha256':vh,'conv_weight_sha256':ch,'test_opened':True,'test_accessed_for_training':False}
    torch.save({'vit':torch.cat(vs),'conv':torch.cat(cs),'labels':torch.tensor([r['label'] for r in records]),
                'records':records,'identity':identity},dst)
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8'); print(json.dumps(identity,indent=2))


if __name__=='__main__': main()
