"""Run only the conditions frozen in STAGE10_FINAL_TEST_PROTOCOL.json."""
import json
from pathlib import Path
import torch

from pretrained_vq import sha256
from qwen_dual_taichi import QwenDualTaiChi
from train_qwen_dual_taichi import batch


@torch.inference_mode()
def evaluate(model,data,indices,snr,kind,seed,variant='correct'):
    model.eval(); predictions=[]; targets=[]; labels=[]; loss=0.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(indices),2):
            chosen=indices[start:start+2]; images,q,a=batch(data,chosen,variant)
            out=model(images,q,a,snr,kind); loss+=float(out.loss)*len(a)
            predictions.extend(model.generate_answers(images,q,snr,kind,max_new_tokens=12)); targets.extend(a)
            labels.extend(int(data['labels'][i]) for i in chosen)
    correct=[p.lower()==t.lower() for p,t in zip(predictions,targets)]
    per_class={str(c):sum(ok for ok,y in zip(correct,labels) if y==c)/sum(y==c for y in labels) for c in range(10)}
    return {'kind':kind,'snr_db':snr,'seed':seed,'variant':variant,'count':len(indices),'loss':loss/len(indices),
            'exact_match':sum(correct)/len(correct),'per_class_accuracy':per_class}


def main():
    root=Path(__file__).resolve().parent; protocol=json.loads((root/'STAGE10_FINAL_TEST_PROTOCOL.json').read_text())
    checkpoint=root/protocol['checkpoint']; data=torch.load(root/'data/stage10_final_test/test.pt',weights_only=True)
    if sha256(checkpoint).upper()!=protocol['checkpoint_sha256']: raise ValueError('Checkpoint no longer matches frozen protocol')
    if data['identity']['protocol_sha256']!=sha256(root/'STAGE10_FINAL_TEST_PROTOCOL.json'): raise ValueError('Protocol/cache mismatch')
    if data['identity']['test_accessed_for_training'] is not False: raise ValueError('Test contamination marker')
    if len(data['labels'])!=protocol['expected_test_images']: raise ValueError('Test count mismatch')
    torch.set_num_threads(4); model=QwenDualTaiChi(r'D:/新建文件夹 (2)/2026-09-21-14-31-36/taichi-tokecom/models/qwen2.5-0.5b-instruct',lora_rank=8)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True); named=dict(model.named_parameters())
    for name,value in ck['trainable_state'].items(): named[name].data.copy_(value)
    indices=list(range(len(data['labels']))); out=root/'runs/stage10_final_test'; out.mkdir(parents=True,exist_ok=True); path=out/'results.jsonl'
    completed={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line); completed[(row['variant'],row['kind'],float(row['snr_db']),int(row['seed']))]=row
    planned=[('correct',kind,float(snr),seed) for kind in protocol['channels'] for snr in protocol['snr_db'] for seed in protocol['seeds']]
    planned += [(c['variant'],c['channel'],float(c['snr_db']),c['seed']) for c in protocol['controls']]
    with path.open('a',encoding='utf-8',buffering=1) as stream:
        for variant,kind,snr,seed in planned:
            key=(variant,kind,snr,seed)
            if key in completed: continue
            row=evaluate(model,data,indices,snr,kind,seed,variant); stream.write(json.dumps(row)+'\n'); print(row,flush=True)
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]; summary=[]
    for kind in protocol['channels']:
        for snr in protocol['snr_db']:
            group=[r for r in rows if r['variant']=='correct' and r['kind']==kind and float(r['snr_db'])==float(snr) and r['seed'] in protocol['seeds']]
            values=torch.tensor([r['exact_match'] for r in group]); losses=torch.tensor([r['loss'] for r in group])
            summary.append({'kind':kind,'snr_db':snr,'seeds':len(group),'exact_match_mean':float(values.mean()),
                            'exact_match_std':float(values.std(unbiased=False)),'loss_mean':float(losses.mean())})
    controls=[r for r in rows if (r['variant'],r['kind'],float(r['snr_db']),r['seed']) in [(c['variant'],c['channel'],float(c['snr_db']),c['seed']) for c in protocol['controls']]]
    final={'protocol_sha256':sha256(root/'STAGE10_FINAL_TEST_PROTOCOL.json'),'checkpoint_step':ck['step'],
           'test_images':len(indices),'test_accessed':True,'used_for_training':False,'summary':summary,'controls':controls}
    (out/'summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8'); (out/'status.json').write_text(json.dumps({'completed':True,'test_images':len(indices),'checkpoint_step':ck['step']}),encoding='utf-8')
    print(json.dumps(final,indent=2))


if __name__=='__main__': main()
