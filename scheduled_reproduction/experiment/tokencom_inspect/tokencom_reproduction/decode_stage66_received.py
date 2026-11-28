"""One geometry-bounded native received-stream decode, launched only as an isolated child."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--case',type=int,required=True);args=p.parse_args()
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads(args.plan.read_text(encoding='utf-8'));case=plan['cases'][args.case]
assert case['case_index']==args.case and plan['child_sha256']==sha(__file__)
path=r/case['received_path'];assert sha(path)==case['received_sha256'] and path.stat().st_size<=65536
from cheng_stream_container import unpack
parsed=unpack(path.read_bytes());assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256 and tuple(parsed['shape'])==(4,4)
from cheng_runtime import activate
activate()
import torch
import numpy as np
from PIL import Image
from compressai.zoo.image import model_architectures
torch.set_num_threads(2);checkpoint=r/plan['checkpoint_path'];assert sha(checkpoint)==plan['checkpoint_sha256']
state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
with torch.inference_mode():pixels=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
assert pixels.shape==(1,3,256,256) and np.isfinite(pixels).all() and pixels.min()>=0 and pixels.max()<=1
image=r/plan['original_image_path'];assert sha(image)==plan['original_image_sha256']
with Image.open(image) as source:original=np.asarray(source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
score=-10*math.log10(float(np.mean(np.square(pixels-original),dtype=np.float64)))
result=dict(case_index=args.case,plan_sha256=sha(args.plan),received_sha256=case['received_sha256'],checkpoint_sha256=plan['checkpoint_sha256'],
    pixel_sha256=hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest(),psnr_decoded=score,encoder_called=False,child_sha256=sha(__file__))
out=r/case['child_result_path'];out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
