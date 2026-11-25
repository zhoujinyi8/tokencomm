"""Run frozen holdout recovery, atomically saving per-case tokens without tuning."""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from pretrained_vq import sha256
from evaluate_stage9_pretrained_maskgit import IMAGENET_CLASS_IDS

root=Path(__file__).resolve().parent
run=root/'runs';data=root/'data/imagenette_stage44_holdout'
protocol_path=run/'stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(data/'test.jsonl')==manifest['test_jsonl_sha256']
records=[json.loads(s) for s in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
identity={'protocol_sha256':sha256(protocol_path),'runner_sha256':sha256(Path(__file__)),
          'encoder_lock_sha256':sha256(run/'stage45_encoder_lock.json'),
          'phy_lock_sha256':sha256(run/'stage45_phy_lock.json'),
          'purpose':'frozen evaluation; partial files are not tuning data'}
lock=run/'stage46_recovery_lock.json'
if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))==identity
else:lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
for name,digest in protocol['weight_sha256'].items():assert sha256(root/name)==digest
official=root.parent.parent/'tokencom_official_reference'
sys.path.insert(0,str(official))
from Network.Taming.models.vqgan import VQModel
from Network.transformer import MaskTransformer
from Trainer.vit import MaskGIT
torch.set_num_threads(4)
cfg=OmegaConf.load(official/'pretrained_maskgit/VQGAN/model.yaml')
vq=VQModel(**cfg.model.params).eval()
assert not vq.load_state_dict(torch.load(root/'pretrained/official_tokcom/last.ckpt',map_location='cpu',weights_only=True)['state_dict'],strict=False).missing_keys
model=MaskTransformer(img_size=256,hidden_dim=768,codebook_size=1024,depth=24,heads=16,mlp_dim=3072,dropout=.1).eval()
state=torch.load(root/'pretrained/official_tokcom/MaskGIT_ImageNet_256.pth',map_location='cpu',weights_only=True,mmap=True)['model_state_dict']
model.load_state_dict(state,strict=True,assign=True);del state
wrapper=SimpleNamespace(vit=model,ae=vq,patch_size=16,codebook_size=1024,args=SimpleNamespace(device='cpu'))
wrapper.adap_sche=lambda step,mode:MaskGIT.adap_sche(wrapper,step,mode=mode)
captured=[];decode=vq.decode_code
def capture(codes):
    captured.append(codes.detach().clone())
    return decode(codes)
vq.decode_code=capture
target=run/'stage46_holdout_recovery';target.mkdir(exist_ok=True)
deadline=time.monotonic()+43*60
for image,record in enumerate(records):
    with np.load(run/f'stage45_holdout_tokens/image{image:03d}.npz') as x:
        assert x['source_sha256'].item()==record['sha256']
        tx=torch.from_numpy(x['tokens'].copy()).long().reshape(1,16,16)
    cls=manifest['class_names'][record['label']]
    for trial in range(5):
        out=target/f'image{image:03d}_trial{trial}.npz'
        if out.exists():
            with np.load(out) as x:assert x['source_sha256'].item()==record['sha256']
            continue
        if time.monotonic()>deadline:
            print('TIME_BUDGET_REACHED; resume missing frozen cases',flush=True)
            raise SystemExit(0)
        phy_path=run/f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz'
        with np.load(phy_path) as x:
            assert x['source_sha256'].item()==record['sha256']
            rx=torch.from_numpy(x['received'].copy()).long().reshape_as(tx)
            oracle=torch.from_numpy(x['oracle_token_mask'].copy()).bool().reshape_as(tx)
            crc=torch.from_numpy(x['crc_token_mask'].copy()).bool().reshape_as(tx)
            llr=x['coded_llr'].copy();bits=x['reencoded_bits'].copy()
        results={};steps={}
        with torch.inference_mode():
            for name,mask in [('oracle',oracle),('crc',crc)]:
                if not mask.any():
                    results[name]=rx.numpy().reshape(256).copy();steps[name]=0
                    continue
                torch.manual_seed(protocol['generation_seed']);np.random.seed(protocol['generation_seed'])
                masked=rx.clone();masked[mask]=1024;captured.clear()
                _,_,_,_,history,_,_=MaskGIT.GenDecoding(wrapper,init_code=masked,_mask=mask.clone(),orig_code=tx,
                    nb_sample=1,labels=torch.tensor([IMAGENET_CLASS_IDS[cls]]),**protocol['author_generation'])
                assert len(captured)==3
                final=captured[0].reshape_as(tx)
                assert torch.equal(captured[1].reshape_as(tx),tx)
                assert torch.equal(final[~mask],rx[~mask])
                results[name]=final.numpy().reshape(256).copy();steps[name]=len(history)
        # Risk score and candidate selection are identical to frozen development rule.
        cost=np.logaddexp(0,-(2*bits-1)*llr)[:,:80].reshape(64,4,20).mean(-1).T.reshape(256)
        ids=np.flatnonzero(crc.numpy().reshape(256))
        ordered=ids[np.argsort(-cost[ids],kind='stable')]
        direct=rx.numpy().reshape(256)
        for fraction in protocol['exploratory_paths']['soft_path_cost_gate_fractions']:
            selected=ordered[:max(1,int(np.ceil(len(ids)*fraction)))] if len(ids) else ordered
            final=direct.copy();final[selected]=results['crc'][selected]
            results[f'soft_gate_{fraction}']=final
        temporary=out.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**results,direct=direct,source_sha256=record['sha256'],
                            source_phy_sha256=sha256(phy_path),steps=json.dumps(steps),
                            recovery_lock_sha256=sha256(lock))
        os.replace(temporary,out)
        print(f'completed image={image+1}/100 trial={trial+1}/5',flush=True)
print('complete 500/500; metric aggregation is separate',flush=True)
