"""Evaluate all matched-padded ARQ endpoints under the frozen VQGAN metric convention."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes',type=float,default=12)
args=parser.parse_args();assert 0<args.max_minutes<=30
deadline=time.monotonic()+60*args.max_minutes
root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan_path=run/'stage61_arq_metric_plan.json';plan=read(plan_path.name);source_audit=read('stage61_packet_arq_audit.json')
assert source_audit['all_checks_passed'] and source_audit['complete'] and source_audit['completed_frame_experiments']==300
assert source_audit['plan_sha256']==plan['arq_plan_sha256']==sha(run/'stage61_packet_arq_plan.json')
for name,digest in plan['input_sha256'].items():assert sha(root/name)==digest
assert sha(run/'stage46_environment.json')==plan['author_environment_sha256']
assert sha(run/'stage47_metrics_lock.json')==plan['metric_lock_sha256']
assert sha(run/'stage59_development_recovery_audit.json')==plan['baseline_complete_audit_sha256']
environment=read('stage46_environment.json')
assert {n:importlib.metadata.version(n) for n in environment['packages']}==environment['packages']
official=root.parents[1]/'tokencom_official_reference'
assert all(sha(official/n)==h for n,h in environment['author_source_sha256'].items())
metric_lock=read('stage47_metrics_lock.json')
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric_lock['alexnet_sha256']
identity=dict(script_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),source_arq_audit_sha256=sha(run/'stage61_packet_arq_audit.json'),
              author_environment_sha256=plan['author_environment_sha256'],metric_lock_sha256=plan['metric_lock_sha256'])
lock_path=run/'stage61_arq_metric_lock.json'
if lock_path.exists():assert read(lock_path.name)==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
torch.set_num_threads(4)
cfg=OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml');vq=VQModel(**cfg.model.params).eval()
state=torch.load(root/'pretrained/official_tokcom/last.ckpt',map_location='cpu',weights_only=True)
assert not vq.load_state_dict(state['state_dict'],strict=False).missing_keys;del state
perceptual=lpips.LPIPS(net='alex').eval();clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as saved:sent=saved['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as saved:selected=json.loads(saved['metadata'].item())
data=root/'data/imagenette_stage5';cache=torch.load(data/'token_cache/val.pt',map_location='cpu',weights_only=True)
target=run/'stage61_arq_metrics';target.mkdir(exist_ok=True);calls=0;budget=False
with torch.inference_mode():
    for image in range(10):
        record=cache['records'][selected['selected_indices'][image]];path=data/record['image']
        assert sha(path)==record['sha256']==selected['rows'][image]['image_sha256']
        with Image.open(path) as opened:original=transform(opened.convert('RGB'))[None]
        reference=clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        tx=torch.from_numpy(sent[image].copy()).long().reshape(1,16,16)
        def measure(ids):
            global calls
            calls+=1;output=vq.decode_code(ids)
            mse=float((output-original).square().mean())
            distance=float(perceptual(original,output.clamp(0,1),normalize=True).mean())
            features=clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0,1)))[None])
            correct=int((ids==tx).sum())
            row=dict(correct_tokens=correct,whole_image_ter=1-correct/256,psnr_raw=-10*math.log10(mse),
                     lpips_normalized=distance,clip_clipped=float(torch.nn.functional.cosine_similarity(reference,features).item()))
            assert all(math.isfinite(v) for v in row.values());return row
        clean=measure(tx)
        for trial in range(5):
            for si,snr in enumerate(plan['snr_db']):
                stem=f'snr{si}_image{image:02d}_trial{trial}';out=target/f'{stem}.json'
                source=run/'stage61_packet_arq'/f'{stem}.json';arrays=source.with_suffix('.npz')
                assert sha(source)==source_audit['case_sha256'][source.name]
                case=json.loads(source.read_text(encoding='utf-8'));assert sha(arrays)==case['trace_arrays_sha256']
                if out.exists():
                    previous=json.loads(out.read_text(encoding='utf-8'))
                    assert previous['lock_sha256']==sha(lock_path) and previous['source_arq_case_sha256']==sha(source)
                    continue
                if time.monotonic()>deadline:budget=True;break
                rows=[]
                with np.load(arrays) as saved:
                    for policy in plan['paths']:
                        ids=torch.from_numpy(saved[f'final_{policy}'].copy()).long().reshape_as(tx)
                        if torch.equal(ids,tx):values=dict(clean);origin='exact_clean_id_reference'
                        else:values=measure(ids);origin='decoded_endpoint'
                        assert 256-values['correct_tokens']==case['policies'][policy]['final_token_errors']
                        rows.append(dict(path=policy,metric_source=origin,**values))
                result=dict(image_index=image,trial=trial,snr_index=si,snr_db=snr,source_image_sha256=record['sha256'],
                            source_arq_case_sha256=sha(source),source_arq_arrays_sha256=sha(arrays),lock_sha256=sha(lock_path),rows=rows)
                temporary=out.with_suffix('.partial.json');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(temporary,out)
                print(f'Padded metrics snr={snr} image={image+1}/10 trial={trial+1}/5',flush=True)
            if budget:break
        if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json'))
progress=dict(identity,lock_sha256=sha(lock_path),completed_frames=len(files),expected_frames=300,complete=len(files)==300,
              decoder_forwards_this_invocation=calls,metric_rows=3*len(files),budget_reached=budget,
              exact_clean_reference_rows=sum(row['metric_source']=='exact_clean_id_reference' for p in files
                  for row in json.loads(p.read_text(encoding='utf-8'))['rows']),case_sha256={p.name:sha(p) for p in files})
(run/'stage61_arq_metric_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in progress.items() if k!='case_sha256'},indent=2))
