"""Freeze an engineering diagnostic before loading any stock model for evaluation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
r=Path(__file__).resolve().parent;run=r/'runs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
smoke=json.loads((run/'stage65_backend_smoke.json').read_text(encoding='utf-8'));assert smoke['passed']==8 and smoke['failed']==0
weights=json.loads((run/'stage65_stock_weights.json').read_text(encoding='utf-8'));assert len(weights)==2
for w in weights:assert sha(r/'pretrained/cheng_stock'/w['name'])==w['sha256']
with np.load(run/'stage26_official_demo_tokens.npz') as a:selection=json.loads(a['metadata'].item())
data=r/'data/imagenette_stage5';cache=torch.load(data/'token_cache/val.pt',map_location='cpu',weights_only=True)
images=[]
for i,index in enumerate(selection['selected_indices']):
    record=cache['records'][index];assert sha(data/record['image'])==record['sha256']==selection['rows'][i]['image_sha256']
    images.append(dict(image_index=i,path=str((data/record['image']).relative_to(r)),sha256=record['sha256'],class_name=selection['rows'][i]['class_name']))
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Stock Cheng clean entropy-coding engineering diagnostic, NOT the paper fine-tuned baseline',
    paper_source='https://arxiv.org/html/2502.12096v5#S6',paper_target_bpp=.04,diagnostic_rate_tolerance_relative=.1,
    paper_finetuned_weight_identified=False,paper_100class_manifest_identified=False,
    models=['cheng2020-anchor','cheng2020-attn'],quality=1,training_metric='mse',weights=weights,images=images,
    image_size=[256,256],preprocess='RGB -> Resize((256,256)) -> ToTensor [0,1]; no input normalization',
    expected_cases=20,container_schema='CH65 v1: fixed 34-byte header plus both complete entropy streams; payload and total bpp recorded separately',
    entropy_coder='native upstream rANS',coding_device='CPU',torch_threads=4,
    metrics='PSNR of actual decompressed clipped [0,1] pixels; normalized LPIPS and clipped-PIL official CLIP preprocessing',
    input_selection_sha256=sha(run/'stage26_official_demo_tokens.npz'),metric_lock_sha256=sha(run/'stage47_metrics_lock.json'),
    backend_smoke_sha256=sha(run/'stage65_backend_smoke.json'),native_build_sha256=sha(run/'stage65_native_build.json'),
    dependency_receipt_sha256=sha(run/'stage65_dependency_wheels.json'),
    container_helper_sha256=sha(r/'cheng_stream_container.py'),container_test_sha256=sha(r/'test_cheng_stream_container.py'),
    fresh_process_checks=[dict(model=m,image_index=0) for m in ('cheng2020-anchor','cheng2020-attn')],
    no_training=True,no_channel_simulation=True,no_corrupted_entropy_decoding=True,
    no_heldout_images=True,no_model_or_quality_selection=True,no_bitrate_claim_from_quality_parameter=True,
    existing_studies_changed=False,full_paper_reproduction_complete=False)
p=run/'stage65_cheng_diagnostic_plan.json';assert not p.exists(),'Do not overwrite a frozen plan'
p.write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(dict(plan_sha256=sha(p),images=len(images),expected_cases=20)))
