"""Train cached pretrained-vision TaiChi bridge. Validation only, not full paper reproduction."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from pretrained_taichi import FeatureTaiChi,load_feature_taichi
from run import encode,pad,decode


def get_batch(data,indices,variant='correct'):
    indices=torch.as_tensor(indices)
    low=data['lr'][indices];high=data['hr'][indices]
    if variant=='zero':low=torch.zeros_like(low);high=torch.zeros_like(high)
    if variant=='wrong_class':
        donors=torch.tensor([int((data['labels']==(int(c)+1)%10).nonzero()[0]) for c in data['labels'][indices]])
        low=data['lr'][donors];high=data['hr'][donors]
    q=pad([encode(data['records'][int(i)]['question']) for i in indices],'cpu')
    a=pad([[1]+encode(data['records'][int(i)]['answer'])+[2] for i in indices],'cpu')
    return (low,high),q,a


@torch.inference_mode()
def evaluate(model,data,kind='awgn',snr=10.,variant='correct',seed=7300):
    model.eval();total=0.;count=0;pred=[]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),8):
            images,q,a=get_batch(data,range(start,min(start+8,len(data['labels']))),variant)
            logits=model(images,q,a[:,:-1],snr,kind)
            total+=float(F.cross_entropy(logits.flatten(0,1),a[:,1:].reshape(-1),ignore_index=0,reduction='sum'));count+=int(a[:,1:].ne(0).sum())
        torch.manual_seed(seed)
        for start in range(0,len(data['labels']),8):
            images,q,a=get_batch(data,range(start,min(start+8,len(data['labels']))),variant)
            pred.extend(decode(x) for x in model.generate(images,q,snr,kind,max_new_tokens=24).tolist())
    targets=[r['answer'] for r in data['records']]
    return {'exact_match':sum(x==y for x,y in zip(pred,targets))/len(targets),'ce':total/count,
            'predictions':pred,'targets':targets,'channel':kind,'snr_db':snr,'variant':variant,'seed':seed}


def linear_baseline(train,val,out):
    torch.manual_seed(7310); model=torch.nn.Linear(512,10)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    x=train['lr'].mean((2,3));v=val['lr'].mean((2,3))
    for _ in range(300):
        opt.zero_grad();loss=F.cross_entropy(model(x),train['labels']);loss.backward();opt.step()
    with torch.no_grad():
        result={'train_accuracy':float((model(x).argmax(-1)==train['labels']).float().mean()),
                'validation_accuracy':float((model(v).argmax(-1)==val['labels']).float().mean()),
                'steps':300,'scope':'frozen low-resolution features + linear classifier; no communication/answer decoder'}
    torch.save({'state':model.state_dict(),'identity':train['identity']},out/'linear_probe.pt')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--out',default='runs/stage3_pretrained_taichi')
    p.add_argument('--data',default='data/imagenette_pilot/resnet18_features')
    p.add_argument('--init',help='Warm start an existing taichi_resnet18_features checkpoint')
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(7301)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():p.error('Choose a new output directory')
    root=Path(a.data)
    train=torch.load(root/'train.pt',weights_only=True);val=torch.load(root/'val.pt',weights_only=True)
    assert train['identity']==val['identity']
    if a.init:
        init_ck=torch.load(a.init,map_location='cpu',weights_only=True);model=load_feature_taichi(init_ck)
    else:
        model=FeatureTaiChi();model.initialize_language_and_fusion(torch.load('runs/taichi_imagenette/best.pt',map_location='cpu',weights_only=True))
    groups=[(train['labels']==c).nonzero().flatten() for c in range(10)]
    config={'steps':a.steps,'alignment_steps':200,'lr':.0005,'batch_size':20,'balanced_sampling':'2/class per update',
            'seed':7301,'train_images':len(train['labels']),'validation_images':len(val['labels']),'test_accessed':False,
            'identity':train['identity'],'pretrained_backbone_parameters_frozen':True,
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'architecture':'shared frozen ResNet18 dual resolution + adapters + BAN/KAN + tiny language/channel modules',
            'not_paper_clip_llm':True,'checkpoint_selection':'max validation exact match; then min response CE, fixed AWGN10 seed7300'}
    config['initialization']=a.init or 'runs/taichi_imagenette/best.pt language/fusion weights; new adapters'
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    baseline=linear_baseline(train,val,out);torch.manual_seed(7301)
    (out/'linear_baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    print('Frozen visual linear baseline:',baseline,flush=True)
    initial=evaluate(model,val);(out/'initial_validation.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    best_key=(initial['exact_match'],-initial['ce']);best_step=0
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00005)
    def checkpoint(step):
        return {'model':'taichi_resnet18_features','state':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                'rng_cpu':torch.get_rng_state(),'step':step,'best_step':best_step,'feature_identity':train['identity'],'config':config}
    torch.save(checkpoint(0),out/'best.pt');start=time.monotonic()
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('lr_adapter.','hr_adapter.','ban.','kan.')))
    with (out/'history.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for step in range(1,a.steps+1):
            if step==201:model.requires_grad_(True)
            model.train();inds=torch.cat([g[torch.randint(len(g),(2,))] for g in groups])
            images,q,ans=get_batch(train,inds);snr=120. if step<=200 else float(torch.rand(())*15)
            opt.zero_grad(set_to_none=True);logits=model(images,q,ans[:,:-1],snr,'awgn')
            loss=F.cross_entropy(logits.flatten(0,1),ans[:,1:].reshape(-1),ignore_index=0)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();schedule.step()
            row={'step':step,'loss':float(loss.detach()),'snr':snr,'seconds':time.monotonic()-start}
            if step%100==0 or step==a.steps:
                metric=evaluate(model,val);row['validation']={k:v for k,v in metric.items() if k not in ['predictions','targets']}
                key=(metric['exact_match'],-metric['ce'])
                if key>best_key:best_key=key;best_step=step;torch.save(checkpoint(step),out/'best.pt')
                torch.save(checkpoint(step),out/'last.pt');print(row,flush=True)
            f.write(json.dumps(row)+'\n')
    best=load_feature_taichi(torch.load(out/'best.pt',weights_only=True))
    results=[]
    for kind,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
        for variant in ['correct','wrong_class','zero']:
            for seed in [7300,7302]:results.append(evaluate(best,val,kind,snr,variant,seed))
    (out/'results.json').write_text(json.dumps({'best_step':best_step,'linear_probe':baseline,'validation_ablations':results,
        'zero_control':'zero cached feature maps, not ResNet features of a black image','test_accessed':False},indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'completed':True,'best_step':best_step,'test_accessed':False}),encoding='utf-8')
    print('Completed pretrained visual bridge training',flush=True)


if __name__=='__main__':main()
