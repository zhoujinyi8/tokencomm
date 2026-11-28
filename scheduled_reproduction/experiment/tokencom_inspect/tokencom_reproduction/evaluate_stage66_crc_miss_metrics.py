"""Measure the sole corrupt CRC endpoint under the already frozen image metric convention."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from cheng_runtime import activate,sha
from cheng_stream_container import unpack
r,_,_=activate();run=r/'runs';read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_crc_miss_probe_plan.json');probe=read('stage66_crc_miss_probe.json');assert probe['all_checks_passed'] and probe['outcome']=='decoded_finite_pixels'
assert probe['plan_sha256']==sha(run/'stage66_crc_miss_probe_plan.json')
metric=read('stage47_metrics_lock.json');assert sha(run/'stage47_metrics_lock.json')==read('stage66_holdout_plan.json')['metric_lock_sha256']
identity=dict(script_sha256=sha(__file__),probe_result_sha256=sha(run/'stage66_crc_miss_probe.json'),probe_plan_sha256=sha(run/'stage66_crc_miss_probe_plan.json'),
    metric_lock_sha256=sha(run/'stage47_metrics_lock.json'),checkpoint_sha256=plan['checkpoint_sha256'],runtime_helper_sha256=sha(r/'cheng_runtime.py'),decoder_threads=2,perceptual_threads=4)
lock_path=run/'stage66_crc_miss_metrics_lock_v2.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
import clip,lpips
from PIL import Image
from torchvision import transforms
from compressai.zoo.image import model_architectures
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric['alexnet_sha256']
torch.set_num_threads(2);checkpoint=r/plan['checkpoint_path'];assert sha(checkpoint)==plan['checkpoint_sha256']
state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
case=plan['cases'][0];stream=r/case['received_path'];assert sha(stream)==case['received_sha256'];parsed=unpack(stream.read_bytes())
assert parsed['height']==parsed['width']==256 and tuple(parsed['shape'])==(4,4)
image=r/plan['original_image_path'];assert sha(image)==plan['original_image_sha256']
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
with Image.open(image) as source:x=transform(source.convert('RGB'))[None]
perceptual=lpips.LPIPS(net='alex').eval();clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
with torch.inference_mode():
    decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'];pixels=decoded.cpu().numpy()
    pixel_sha=hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest();assert pixel_sha==probe['metrics']['pixel_sha256']
    torch.set_num_threads(4)
    psnr=-10*math.log10(float((decoded-x).square().mean()));assert abs(psnr-probe['metrics']['psnr_decoded'])<1e-4
    distance=float(perceptual(x,decoded,normalize=True).mean())
    a=clip_model.encode_image(clip_preprocess(to_pil(x[0]))[None]);b=clip_model.encode_image(clip_preprocess(to_pil(decoded[0]))[None])
    similarity=float(torch.nn.functional.cosine_similarity(a,b).item())
array=run/'stage66_crc_miss_probe/metrics_pixels.npz';np.savez_compressed(array,pixels=pixels)
value=dict(all_checks_passed=True,lock_sha256=sha(lock_path),image_index=case['image_index'],trial=case['trial'],snr_db=case['snr_db'],
    received_container_sha256=case['received_sha256'],checkpoint_sha256=plan['checkpoint_sha256'],pixel_sha256=pixel_sha,reconstruction_sha256=sha(array),
    psnr_decoded=psnr,lpips_normalized=distance,clip_clipped=similarity,new_native_decoder_forwards=1,encoder_called=False,paper_baseline_ready=False)
assert all(math.isfinite(value[k]) for k in ('psnr_decoded','lpips_normalized','clip_clipped'))
(run/'stage66_crc_miss_metrics.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))
