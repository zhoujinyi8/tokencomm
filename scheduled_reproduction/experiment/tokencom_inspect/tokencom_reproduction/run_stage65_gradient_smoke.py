import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from cheng_runtime import activate,load_stock,sha
r,source,_=activate();run=r/'runs';plan=json.loads((run/'stage65_gradient_smoke_plan.json').read_text(encoding='utf-8'))
receipt=run/'stage65_gradient_smoke.json';assert not receipt.exists(),'Do not repeat the optimizer-update experiment'
assert plan['steps']==1 and plan['paper_baseline_ready'] is False
for name,h in plan['evaluation_manifest_sha256'].items():assert sha(r/name)==h
path=r/plan['image']['path'];assert sha(path)==plan['image']['sha256']
torch.set_num_threads(plan['torch_threads']);weight=r/'pretrained/cheng_stock/cheng2020-anchor-1-dad2ebff.pth.tar'
assert sha(weight)==plan['stock_weight_sha256'];model=load_stock('cheng2020-anchor',weight)
torch.manual_seed(plan['seed']);np.random.seed(plan['seed']%(2**32))
with Image.open(path) as image:x=transforms.ToTensor()(transforms.Resize((256,256))(image.convert('RGB')))[None]
parameters=dict(model.named_parameters());main={n:p for n,p in parameters.items() if p.requires_grad and not n.endswith('.quantiles')}
aux={n:p for n,p in parameters.items() if p.requires_grad and n.endswith('.quantiles')}
assert main and aux and not set(main)&set(aux)
before={n:p.detach().clone() for n,p in main.items()}
optimizer=torch.optim.Adam(main.values(),lr=plan['learning_rate']);aux_optimizer=torch.optim.Adam(aux.values(),lr=plan['aux_learning_rate'])
from compressai.losses import RateDistortionLoss
criterion=RateDistortionLoss(lmbda=plan['lmbda'],metric='mse');model.train()
optimizer.zero_grad(set_to_none=True);aux_optimizer.zero_grad(set_to_none=True)
loss=criterion(model(x),x);assert all(torch.isfinite(v).all() for v in loss.values());loss['loss'].backward()
gradients=[p.grad for p in main.values() if p.grad is not None];assert gradients and all(torch.isfinite(g).all() for g in gradients)
norm=float(torch.nn.utils.clip_grad_norm_(list(main.values()),plan['max_grad_norm']));assert math.isfinite(norm)
optimizer.step();aux_loss=model.aux_loss();assert torch.isfinite(aux_loss).all();aux_loss.backward();aux_optimizer.step()
changed=sum(not torch.equal(before[n],p.detach()) for n,p in main.items());assert changed>0
model.eval();model.update(force=True)
target=r/'pretrained/cheng_smoke';target.mkdir(exist_ok=True);checkpoint=target/'anchor_one_update.pt';assert not checkpoint.exists()
torch.save(dict(state_dict=model.state_dict(),plan_sha256=sha(run/'stage65_gradient_smoke_plan.json'),optimizer_updates=1),checkpoint)
from compressai.zoo.image import model_architectures
restored=torch.load(checkpoint,map_location='cpu',weights_only=True);assert restored['plan_sha256']==sha(run/'stage65_gradient_smoke_plan.json')
fresh=model_architectures['cheng2020-anchor'].from_state_dict(restored['state_dict']).eval();fresh.update(force=True)
with torch.inference_mode():
    encoded=model.compress(x);a=model.decompress(encoded['strings'],encoded['shape'])['x_hat'];b=fresh.decompress(encoded['strings'],encoded['shape'])['x_hat']
error=float((a-b).abs().max());assert error<=1e-6 and torch.isfinite(a).all()
assert sha(weight)==plan['stock_weight_sha256']
result=dict(all_checks_passed=True,script_sha256=sha(__file__),plan_sha256=sha(run/'stage65_gradient_smoke_plan.json'),
    runtime_helper_sha256=sha(r/'cheng_runtime.py'),optimizer_updates=1,main_parameter_tensors=len(main),changed_parameter_tensors=changed,
    finite_gradient_tensors=len(gradients),gradient_norm_before_clip=norm,training_loss={k:float(v.detach()) for k,v in loss.items()},aux_loss=float(aux_loss.detach()),
    checkpoint_path=str(checkpoint.relative_to(r)),checkpoint_sha256=sha(checkpoint),reloaded_decode_max_abs_difference=error,
    image_sha256=sha(path),stock_weights_unchanged=True,frozen_evaluation_images_used_for_training=False,
    rate_match_established=False,paper_baseline_ready=False,full_paper_reproduction_complete=False)
receipt.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
