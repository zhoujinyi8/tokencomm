"""All-arm clean-development LPIPS/CLIP metrics under the existing frozen metric convention."""
import hashlib,importlib.metadata,json,math,os
from pathlib import Path
import numpy as np
import torch
from cheng_runtime import activate,sha
from cheng_stream_container import unpack
r,_,_=activate();run=r/'runs';read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_rate_grid_plan.json');audit=read('stage66_rate_grid_audit.json');grid=read('stage66_rate_grid.json')
assert audit['all_checks_passed'] and audit['complete'] and audit['source_result_sha256']==sha(run/'stage66_rate_grid.json')
assert audit['plan_sha256']==sha(run/'stage66_rate_grid_plan.json')
metric_lock=read('stage47_metrics_lock.json');diagnostic=read('stage65_cheng_diagnostic_plan.json')
assert sha(run/'stage47_metrics_lock.json')==diagnostic['metric_lock_sha256']
import clip,lpips
from PIL import Image
from torchvision import transforms
from compressai.zoo.image import model_architectures
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric_lock['alexnet_sha256']
identity=dict(script_sha256=sha(__file__),plan_sha256=audit['plan_sha256'],grid_audit_sha256=sha(run/'stage66_rate_grid_audit.json'),
              grid_result_sha256=sha(run/'stage66_rate_grid.json'),metric_lock_sha256=sha(run/'stage47_metrics_lock.json'),
              runtime_helper_sha256=sha(r/'cheng_runtime.py'),scope='all three arms, all ten old development images, clean decoded streams')
lock_path=run/'stage66_perceptual_lock.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
torch.set_num_threads(plan['torch_threads']);perceptual=lpips.LPIPS(net='alex').eval()
clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
out_path=run/'stage66_perceptual.json'
if out_path.exists():
    output=read(out_path.name);assert output['lock_sha256']==sha(lock_path)
else:output=dict(lock_sha256=sha(lock_path),cases=[],complete=False,paper_baseline_ready=False,final_test_performed=False)
with torch.inference_mode():
    for arm in grid['arms']:
        path=r/arm['checkpoint_path'];assert sha(path)==arm['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
        for case in arm['calibration_cases']:
            if any(v['arm_id']==arm['arm_id'] and v['image_index']==case['image_index'] for v in output['cases']):continue
            image=next(v for v in plan['calibration_images'] if v['image_index']==case['image_index']);assert sha(r/image['path'])==image['sha256']
            with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
            stream=r/case['container_path'];assert sha(stream)==case['container_sha256'];parsed=unpack(stream.read_bytes())
            decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat']
            assert hashlib.sha256(decoded.cpu().numpy().astype('<f4',copy=False).tobytes()).hexdigest()==case['decoded_pixel_sha256']
            psnr=-10*math.log10(float((decoded-x).square().mean()));assert abs(psnr-case['psnr_decoded'])<1e-4
            distance=float(perceptual(x,decoded,normalize=True).mean())
            a=clip_model.encode_image(clip_preprocess(to_pil(x[0]))[None]);b=clip_model.encode_image(clip_preprocess(to_pil(decoded[0]))[None])
            similarity=float(torch.nn.functional.cosine_similarity(a,b).item())
            assert math.isfinite(distance) and distance>=0 and math.isfinite(similarity) and -1.00001<=similarity<=1.00001
            output['cases'].append(dict(arm_id=arm['arm_id'],image_index=case['image_index'],checkpoint_sha256=arm['checkpoint_sha256'],
                container_sha256=case['container_sha256'],decoded_pixel_sha256=case['decoded_pixel_sha256'],psnr_decoded=psnr,
                lpips_normalized=distance,clip_clipped=similarity,model_encoder_called=False))
            temporary=out_path.with_suffix('.tmp.json');temporary.write_text(json.dumps(output,indent=2),encoding='utf-8');os.replace(temporary,out_path)
            print(f'Metrics {arm["arm_id"]} image={case["image_index"]+1}/10',flush=True)
        del model,state
assert len(output['cases'])==30 and len({(v['arm_id'],v['image_index']) for v in output['cases']})==30
output['complete']=True;out_path.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(dict(complete=True,cases=30,lock_sha256=output['lock_sha256'])))
