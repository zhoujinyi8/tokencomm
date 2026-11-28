"""Fixed repeated corrupt-stream decodes to diagnose a failed pixel-identity preflight."""
import hashlib,json,math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from cheng_runtime import activate,sha
from cheng_stream_container import unpack
r,_,_=activate();run=r/'runs';read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_crc_miss_probe_plan.json');probe=read('stage66_crc_miss_probe.json')
checkpoint=r/plan['checkpoint_path'];assert sha(checkpoint)==plan['checkpoint_sha256']
from compressai.zoo.image import model_architectures
torch.set_num_threads(2);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
case=plan['cases'][0];path=r/case['received_path'];assert sha(path)==case['received_sha256'];parsed=unpack(path.read_bytes())
with Image.open(r/plan['original_image_path']) as source:original=np.asarray(source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
rows=[];reference=None
with torch.inference_mode():
    for index,threads in enumerate((2,2,4,2)):
        torch.set_num_threads(threads);pixels=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
        pixel_hash=hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest()
        if reference is None:reference=pixels.copy()
        row=dict(repetition=index,torch_threads=threads,pixel_sha256=pixel_hash,equals_original_child_hash=pixel_hash==probe['metrics']['pixel_sha256'],
            max_abs_pixel_difference_from_first=float(np.max(np.abs(pixels-reference))),psnr_decoded=-10*math.log10(float(np.mean(np.square(pixels-original),dtype=np.float64))))
        rows.append(row);print(json.dumps(row),flush=True)
value=dict(all_runs_completed=True,script_sha256=sha(__file__),probe_plan_sha256=sha(run/'stage66_crc_miss_probe_plan.json'),received_sha256=case['received_sha256'],
    checkpoint_sha256=plan['checkpoint_sha256'],rows=rows,identity_gate_relaxed=False,cohort_quality_published=False)
(run/'stage66_crc_repeat_diagnostic.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))
