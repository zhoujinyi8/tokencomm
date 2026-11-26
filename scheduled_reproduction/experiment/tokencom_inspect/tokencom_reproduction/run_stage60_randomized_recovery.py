"""Independent preregistered random-packet recovery in canonical token coordinates."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import clip
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes',type=float,default=35)
args=parser.parse_args();assert 0<args.max_minutes<=50
deadline=time.monotonic()+args.max_minutes*60
root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
plan_path=run/'stage60_execution_spec.json'
plan=json.loads(plan_path.read_text(encoding='utf-8'))
assert sha(run/'stage59_randomized_packetization_plan.json')==plan['preregistered_plan_sha256']
assert sha(run/'stage59_development_recovery_audit.json')==plan['baseline_complete_audit_sha256']
assert sha(root/'evaluate_stage9_pretrained_maskgit.py')==plan['class_mapping_script_sha256']
assert sha(root/'randomized_packetization.py')==plan['packetization_helper_sha256']
contract=json.loads((run/'stage60_packetization_contract.json').read_text(encoding='utf-8'))
assert contract['passed']==22 and contract['failed']==0
assert contract['helper_sha256']==plan['packetization_helper_sha256']
assert contract['test_script_sha256']==sha(root/'test_randomized_packetization.py')
for name,digest in plan['input_sha256'].items():assert sha(root/name)==digest
lock_path=run/'stage60_randomized_recovery_lock.json'
identity={'script_sha256':sha(Path(__file__)),'plan_sha256':sha(plan_path),
          'author_environment_sha256':sha(run/'stage46_environment.json'),
          'metric_lock_sha256':sha(run/'stage47_metrics_lock.json'),
          'token_metric':'final token ID equality; author TER_af is not residual TER',
          'preregistered_plan_sha256':plan['preregistered_plan_sha256'],
          'baseline_complete_audit_sha256':plan['baseline_complete_audit_sha256'],
          'phy_audit_sha256':sha(run/'stage60_randomized_phy_audit.json')}
if lock_path.exists():assert json.loads(lock_path.read_text(encoding='utf-8'))==identity
else:lock_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
phy_audit=json.loads((run/'stage60_randomized_phy_audit.json').read_text(encoding='utf-8'))
assert phy_audit['all_checks_passed'] and phy_audit['complete'] and phy_audit['plan_sha256']==sha(plan_path)
metric_lock=json.loads((run/'stage47_metrics_lock.json').read_text(encoding='utf-8'))
assert importlib.metadata.version('lpips')=='0.1.4'
assert sha(Path.home()/'.cache/clip/ViT-B-32.pt')==metric_lock['clip_weight_sha256']
assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==metric_lock['lpips_linear_sha256']
assert sha(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')==metric_lock['alexnet_sha256']
official=root.parents[1]/'tokencom_official_reference';sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS
torch.set_num_threads(4)
cfg=OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq=VQModel(**cfg.model.params).eval()
state=torch.load(root/'pretrained/official_tokcom/last.ckpt',map_location='cpu',weights_only=True)
assert not vq.load_state_dict(state['state_dict'],strict=False).missing_keys;del state
model=MaskTransformer(img_size=256,hidden_dim=768,codebook_size=1024,depth=24,heads=16,mlp_dim=3072,dropout=.1).eval()
state=torch.load(root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth',map_location='cpu',weights_only=True,mmap=True)['model_state_dict']
model.load_state_dict(state,strict=True,assign=True);del state
wrapper=SimpleNamespace(vit=model,ae=vq,patch_size=16,codebook_size=1024,args=SimpleNamespace(device='cpu'))
wrapper.adap_sche=lambda step,mode:MaskGIT.adap_sche(wrapper,step,mode=mode)
perceptual=lpips.LPIPS(net='alex').eval()
clip_model,clip_preprocess=clip.load('ViT-B/32',device='cpu');clip_model.eval()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]);to_pil=transforms.ToPILImage()
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as link:sent=link['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as source:selected=json.loads(source['metadata'].item())
data=root/'data/imagenette_stage5';cache=torch.load(data/'token_cache/val.pt',map_location='cpu',weights_only=True)
target=run/'stage60_randomized_recovery';target.mkdir(exist_ok=True)
captured=[];decode=vq.decode_code


def capture(codes):
    captured.append(codes.detach().clone())
    return decode(codes)


vq.decode_code=capture
budget=False
with torch.inference_mode():
    for image in range(10):
        record=cache['records'][selected['selected_indices'][image]]
        source=data/record['image'];assert sha(source)==record['sha256']==selected['rows'][image]['image_sha256']
        with Image.open(source) as opened:original=transform(opened.convert('RGB'))[None]
        reference=clip_model.encode_image(clip_preprocess(to_pil(original[0]))[None])
        tx=torch.from_numpy(sent[image].copy()).long().reshape(1,16,16)
        label=IMAGENET_CLASS_IDS[selected['rows'][image]['class_name']]

        def measure(name,output,ids,steps):
            mse=float((output-original).square().mean())
            distance=float(perceptual(original,output.clamp(0,1),normalize=True).mean())
            features=clip_model.encode_image(clip_preprocess(to_pil(output[0].clamp(0,1)))[None])
            cosine=float(torch.nn.functional.cosine_similarity(reference,features).item())
            correct=int((ids==tx).sum())
            row={'path':name,'correct_tokens':correct,'whole_image_ter':1-correct/256,'steps':steps,
                 'psnr_raw':-10*math.log10(mse),'lpips_normalized':distance,'clip_clipped':cosine}
            assert all(math.isfinite(row[key]) for key in ('psnr_raw','lpips_normalized','clip_clipped'))
            return row

        clean_row=measure('ideal_correct_tokens',vq.decode_code(tx),tx,0)
        for trial in range(5):
            for snr_index,snr in enumerate(plan['snr_db']):
                stem=f'snr{snr_index}_image{image:02d}_trial{trial}'
                phy_path=run/'stage60_randomized_phy'/f'{stem}.npz'
                if not phy_path.exists():continue
                out=target/f'{stem}.json';latent_path=target/f'{stem}.npz'
                if out.exists():
                    saved=json.loads(out.read_text(encoding='utf-8'))
                    assert saved['lock_sha256']==sha(lock_path) and saved['source_phy_sha256']==sha(phy_path)
                    assert saved['latent_sha256']==sha(latent_path)
                    continue
                if time.monotonic()>deadline:budget=True;break
                assert sha(phy_path)==phy_audit['case_sha256'][phy_path.name]
                with np.load(phy_path) as phy:
                    assert phy['plan_sha256'].item()==sha(plan_path) and phy['source_sha256'].item()==record['sha256']
                    rx=torch.from_numpy(phy['received'].copy()).long().reshape(1,16,16)
                    mask=torch.from_numpy(phy['oracle_token_mask'].copy()).bool().reshape(1,16,16)
                rows=[measure('direct',vq.decode_code(rx),rx,0)];codes={'direct':rx.numpy().reshape(256).copy()}
                for name,mode in [('without_cmi',0),('with_cmi',1)]:
                    if not mask.any():
                        final=rx;row=dict(rows[0],path=name)
                    else:
                        torch.manual_seed(plan['generation_seed']);np.random.seed(plan['generation_seed'])
                        masked=rx.clone();masked[mask]=1024;captured.clear()
                        generated=MaskGIT.GenDecoding(wrapper,init_code=masked,_mask=mask.clone(),orig_code=tx,
                            nb_sample=1,labels=torch.tensor([label]),w=mode,**plan['generation_parameters'])
                        final=captured[0].reshape_as(tx)
                        assert torch.equal(final[~mask],rx[~mask])
                        correct_masked=int((final[mask]==tx[mask]).sum())
                        assert math.isclose(float(generated[5]),correct_masked/256,abs_tol=1e-8)
                        row=measure(name,generated[0],final,len(generated[4]))
                    rows.append(row);codes[name]=final.numpy().reshape(256).copy()
                rows.append(dict(clean_row));codes['ideal_correct_tokens']=tx.numpy().reshape(256).copy()
                temporary=latent_path.with_suffix('.partial.npz');np.savez_compressed(temporary,**codes);os.replace(temporary,latent_path)
                result={'snr_index':snr_index,'snr_db':snr,'image_index':image,'trial':trial,
                        'source_sha256':record['sha256'],'source_phy_sha256':sha(phy_path),
                        'lock_sha256':sha(lock_path),'latent_sha256':sha(latent_path),'masked_tokens':int(mask.sum()),'rows':rows}
                temporary=out.with_suffix('.partial.json');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');os.replace(temporary,out)
                print(f'Random model snr={snr} image={image+1}/10 trial={trial+1}/5',flush=True)
            if budget:break
        if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].json'))
counts={str(snr):sum(path.name.startswith(f'snr{i}_') for path in files) for i,snr in enumerate(plan['snr_db'])}
result={'scope':plan['scope'],'plan_sha256':sha(plan_path),'lock_sha256':sha(lock_path),
        'completed_cases':len(files),'expected_cases':300,'complete':len(files)==300,'counts_by_snr':counts,
        'budget_reached':budget,'primary_holdout_changed':False,'case_sha256':{path.name:sha(path) for path in files}}
(run/'stage60_randomized_recovery_progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='case_sha256'},indent=2))
