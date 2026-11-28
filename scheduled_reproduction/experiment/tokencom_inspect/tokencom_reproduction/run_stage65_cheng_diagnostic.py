"""Actual stock-Cheng byte streams and clean decoded endpoint metrics, never paper-ready by substitution."""
import argparse,importlib.metadata,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from cheng_runtime import activate,load_stock,sha
from cheng_stream_container import pack,unpack
p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=20);args=p.parse_args();assert 0<args.max_minutes<=30
r,source,compressai=activate();run=r/'runs';deadline=time.monotonic()+60*args.max_minutes
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage65_cheng_diagnostic_plan.json');smoke=read('stage65_backend_smoke.json')
assert smoke['passed']==8 and sha(run/'stage65_backend_smoke.json')==plan['backend_smoke_sha256']
assert smoke['runtime_helper_sha256']==sha(r/'cheng_runtime.py')
for n,h in smoke['source_python_sha256'].items():assert sha(source/n)==h
for n,v in smoke['versions'].items():assert importlib.metadata.version(n)==v
assert sha(r/'cheng_stream_container.py')==plan['container_helper_sha256']
assert sha(r/'test_cheng_stream_container.py')==plan['container_test_sha256']
metric_lock=read('stage47_metrics_lock.json');assert sha(run/'stage47_metrics_lock.json')==plan['metric_lock_sha256']
import clip,lpips
from PIL import Image
from torchvision import transforms
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric_lock['alexnet_sha256']
identity=dict(script_sha256=sha(__file__),plan_sha256=sha(run/'stage65_cheng_diagnostic_plan.json'),
    runtime_helper_sha256=sha(r/'cheng_runtime.py'),container_helper_sha256=sha(r/'cheng_stream_container.py'),
    auditor_sha256=sha(r/'audit_stage65_cheng_diagnostic.py'),fresh_decoder_sha256=sha(r/'decode_cheng_container.py'),
    native_build_sha256=sha(run/'stage65_native_build.json'),backend_smoke_sha256=sha(run/'stage65_backend_smoke.json'))
lock=run/'stage65_cheng_diagnostic_lock.json'
if lock.exists():assert read(lock.name)==identity
else:lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
torch.set_num_threads(plan['torch_threads']);perceptual=lpips.LPIPS(net='alex').eval()
clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
target=run/'stage65_cheng_cases';target.mkdir(exist_ok=True);budget=False
with torch.inference_mode():
    for mi,model_name in enumerate(plan['models']):
        weight=plan['weights'][mi];wp=r/'pretrained/cheng_stock'/weight['name'];assert sha(wp)==weight['sha256']
        model=load_stock(model_name,wp)
        for image in plan['images']:
            i=image['image_index'];stem=f'model{mi}_image{i:02d}';out=target/(stem+'.json')
            if out.exists():
                saved=json.loads(out.read_text(encoding='utf-8'))
                assert saved['lock_sha256']==sha(lock) and sha(target/(stem+'.ch65'))==saved['container_sha256'] and sha(target/(stem+'.npz'))==saved['reconstruction_sha256']
                continue
            if time.monotonic()>deadline:budget=True;break
            path=r/image['path'];assert sha(path)==image['sha256']
            with Image.open(path) as opened:x=transform(opened.convert('RGB'))[None]
            estimated=model(x)
            likelihood_bpp=sum(float(torch.log(v).sum()) for v in estimated['likelihoods'].values())/(-math.log(2)*256*256)
            started=time.monotonic();encoded=model.compress(x)
            assert len(encoded['strings'])==2 and all(len(s)==1 for s in encoded['strings'])
            streams=[s[0] for s in encoded['strings']];shape=tuple(map(int,encoded['shape']))
            blob=pack(model_name,1,256,256,shape,streams);parsed=unpack(blob)
            assert parsed['streams']==streams and parsed['shape']==shape
            decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat']
            elapsed=time.monotonic()-started
            assert decoded.shape==x.shape and torch.isfinite(decoded).all() and decoded.min()>=0 and decoded.max()<=1
            mse=float((decoded-x).square().mean());psnr=-10*math.log10(mse)
            distance=float(perceptual(x,decoded,normalize=True).mean())
            a=clip_model.encode_image(clip_preprocess(to_pil(x[0]))[None]);b=clip_model.encode_image(clip_preprocess(to_pil(decoded[0]))[None])
            similarity=float(torch.nn.functional.cosine_similarity(a,b).item())
            pixels=decoded.cpu().numpy();container=target/(stem+'.ch65');container.write_bytes(blob)
            array=target/(stem+'.npz');temp=array.with_suffix('.partial.npz');np.savez_compressed(temp,pixels=pixels);os.replace(temp,array)
            bits=sum(len(s) for s in streams)*8;header_bits=parsed['header_bytes']*8;total_bits=len(blob)*8
            assert total_bits==bits+header_bits
            row=dict(model=model_name,quality=1,image_index=i,source_image_sha256=image['sha256'],weight_sha256=weight['sha256'],
                lock_sha256=sha(lock),container_sha256=sha(container),reconstruction_sha256=sha(array),decoded_pixels_sha256=__import__('hashlib').sha256(pixels.tobytes()).hexdigest(),
                entropy_stream_bytes=[len(s) for s in streams],shape=list(shape),entropy_payload_bits=bits,container_header_bits=header_bits,total_stored_bits=total_bits,
                entropy_payload_bpp=bits/65536,total_stored_bpp=total_bits/65536,estimated_likelihood_bpp=likelihood_bpp,
                psnr_decoded=psnr,lpips_normalized=distance,clip_clipped=similarity,encode_decode_seconds=elapsed,
                paper_baseline_ready=False,channel_simulated=False)
            assert all(math.isfinite(row[k]) for k in ('entropy_payload_bpp','total_stored_bpp','estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped'))
            temp=out.with_suffix('.partial.json');temp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(temp,out)
            print(f'Clean stock codec {model_name} image={i+1}/10 payload_bpp={bits/65536:.6f} total_bpp={total_bits/65536:.6f}',flush=True)
        del model
        if budget:break
files=sorted(target.glob('model[01]_image[0-9][0-9].json'))
progress=dict(identity,lock_sha256=sha(lock),completed_cases=len(files),expected_cases=20,complete=len(files)==20,budget_reached=budget,
              case_sha256={f.name:sha(f) for f in files},paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage65_cheng_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in progress.items() if k!='case_sha256'},indent=2))
