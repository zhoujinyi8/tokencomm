"""Actual fixed-checkpoint clean held-out codec and image metrics; resumable complete cases only."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from cheng_runtime import activate,sha
from cheng_stream_container import pack,unpack
from cheng_packet_budget import packet_budget
p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=25);args=p.parse_args();assert 0<args.max_minutes<=30
r,_,_=activate();run=r/'runs';read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan_path=run/'stage66_holdout_plan.json';plan=read(plan_path.name)
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
for n,h in plan['historical_evidence_sha256'].items():assert sha(run/n)==h
assert sha(run/'stage66_rate_grid_audit.json')==plan['grid_audit_sha256']
assert sha(run/'stage47_metrics_lock.json')==plan['metric_lock_sha256'];metric=read('stage47_metrics_lock.json')
checkpoint=r/plan['checkpoint_path'];assert sha(checkpoint)==plan['checkpoint_sha256']
import clip,lpips
from PIL import Image
from torchvision import transforms
from compressai.zoo.image import model_architectures
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric['alexnet_sha256']
torch.set_num_threads(plan['torch_threads'])
state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['step']==100 and state['arm']['arm_id']==plan['selected_arm_id']
model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
perceptual=lpips.LPIPS(net='alex').eval();clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
target=run/'stage66_holdout_cases';target.mkdir(exist_ok=True);deadline=time.monotonic()+args.max_minutes*60
with torch.inference_mode():
    for image in plan['images']:
        index=image['image_index'];out=target/f'image{index:03d}.json'
        if out.exists():
            previous=json.loads(out.read_text(encoding='utf-8'));assert previous['plan_sha256']==sha(plan_path)
            assert sha(out.with_suffix('.ch65'))==previous['container_sha256'] and sha(out.with_suffix('.npz'))==previous['reconstruction_sha256']
            continue
        if time.monotonic()>=deadline:break
        assert sha(r/image['path'])==image['sha256']
        with Image.open(r/image['path']) as opened:x=transform(opened.convert('RGB'))[None]
        likelihoods=model(x)['likelihoods'];estimated=sum(float(torch.log(v).sum()) for v in likelihoods.values())/(-math.log(2)*65536)
        encoded=model.compress(x);streams=[s[0] for s in encoded['strings']];shape=tuple(map(int,encoded['shape']))
        blob=pack('cheng2020-anchor',1,256,256,shape,streams);parsed=unpack(blob)
        decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat']
        assert decoded.shape==x.shape and torch.isfinite(decoded).all() and decoded.min()>=0 and decoded.max()<=1
        mse=float((decoded-x).square().mean());psnr=-10*math.log10(mse);distance=float(perceptual(x,decoded,normalize=True).mean())
        a=clip_model.encode_image(clip_preprocess(to_pil(x[0]))[None]);b=clip_model.encode_image(clip_preprocess(to_pil(decoded[0]))[None])
        similarity=float(torch.nn.functional.cosine_similarity(a,b).item());pixels=decoded.cpu().numpy()
        container=out.with_suffix('.ch65');container.write_bytes(blob)
        array=out.with_suffix('.npz');temporary=array.with_suffix('.partial.npz');np.savez_compressed(temporary,pixels=pixels);os.replace(temporary,array)
        row=dict(image_index=index,source_image_sha256=image['sha256'],checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha(plan_path),
            container_sha256=sha(container),reconstruction_sha256=sha(array),decoded_pixels_sha256=hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest(),
            entropy_payload_bpp=sum(len(s) for s in streams)*8/65536,total_stored_bpp=len(blob)*8/65536,estimated_likelihood_bpp=estimated,
            psnr_decoded=psnr,lpips_normalized=distance,clip_clipped=similarity,packet_budget=packet_budget(len(blob)),
            channel_simulated=False,paper_baseline_ready=False)
        assert all(math.isfinite(row[k]) for k in ('entropy_payload_bpp','total_stored_bpp','estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped'))
        temporary=out.with_suffix('.tmp.json');temporary.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(temporary,out)
        print(f'Frozen Cheng holdout image={index+1}/100 entropy_bpp={row["entropy_payload_bpp"]:.6f}',flush=True)
files=sorted(target.glob('image[0-9][0-9][0-9].json'))
progress=dict(plan_sha256=sha(plan_path),worker_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),completed_cases=len(files),expected_cases=100,
    complete=len(files)==100,case_sha256={f.name:sha(f) for f in files},paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage66_holdout_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in progress.items() if k!='case_sha256'}))
