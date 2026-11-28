"""Fresh-process decoding from a container and pinned stock checkpoint only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from cheng_runtime import activate,load_stock,sha
from cheng_stream_container import unpack
p=argparse.ArgumentParser();p.add_argument('--model-index',type=int,required=True);a=p.parse_args();assert a.model_index in (0,1)
r,source,_=activate();run=r/'runs';plan=json.loads((run/'stage65_cheng_diagnostic_plan.json').read_text(encoding='utf-8'))
lock=json.loads((run/'stage65_cheng_diagnostic_lock.json').read_text(encoding='utf-8'));assert lock['fresh_decoder_sha256']==sha(__file__)
assert lock['plan_sha256']==sha(run/'stage65_cheng_diagnostic_plan.json') and lock['runtime_helper_sha256']==sha(r/'cheng_runtime.py')
torch.set_num_threads(plan['torch_threads']);model_name=plan['models'][a.model_index];weight=plan['weights'][a.model_index]
wp=r/'pretrained/cheng_stock'/weight['name'];assert sha(wp)==weight['sha256']
model=load_stock(model_name,wp);stem=f'model{a.model_index}_image00';container=run/'stage65_cheng_cases'/(stem+'.ch65')
parsed=unpack(container.read_bytes());assert parsed['model']==model_name and parsed['quality']==1
with torch.inference_mode():pixels=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
assert pixels.shape==(1,3,256,256) and np.isfinite(pixels).all()
out=run/(f'stage65_fresh_model{a.model_index}.npz');np.savez_compressed(out,pixels=pixels)
receipt=dict(model=model_name,image_index=0,container_sha256=sha(container),weight_sha256=weight['sha256'],
    decoder_script_sha256=sha(__file__),output_sha256=sha(out),decoded_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
    source_image_loaded=False,encoder_called=False,full_paper_reproduction_complete=False)
(run/f'stage65_fresh_model{a.model_index}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
