"""Audit all stored held-out pixels/rates and five preregistered fresh decoder checks."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from cheng_stream_container import unpack
from cheng_packet_budget import packet_budget,packetize,reassemble
p=argparse.ArgumentParser();p.add_argument('--verify-codec',action='store_true');args=p.parse_args()
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage66_holdout_plan.json');progress=read('stage66_holdout_progress.json');target=run/'stage66_holdout_cases'
assert progress['plan_sha256']==sha(run/'stage66_holdout_plan.json') and progress['worker_sha256']==sha(r/'run_stage66_holdout.py')
for n,h in plan['source_sha256'].items():assert sha(r/n)==h
for n,h in plan['historical_evidence_sha256'].items():assert sha(run/n)==h
assert sha(run/'stage66_rate_grid_audit.json')==plan['grid_audit_sha256']
assert sha(run/'stage47_metrics_lock.json')==plan['metric_lock_sha256']
assert sha(r/'data/imagenette_stage44_holdout/test.jsonl')==plan['holdout_manifest_sha256']
checkpoint=r/plan['checkpoint_path'];assert sha(checkpoint)==progress['checkpoint_sha256']==plan['checkpoint_sha256']
model=None
if args.verify_codec:
    import torch
    from cheng_runtime import activate
    activate();torch.set_num_threads(plan['torch_threads'])
    from compressai.zoo.image import model_architectures
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=model_architectures['cheng2020-anchor'].from_state_dict(state['state_dict']).eval();model.update(force=True)
rows=[];fresh=[]
for name,h in progress['case_sha256'].items():
    path=target/name;assert sha(path)==h;row=json.loads(path.read_text(encoding='utf-8'));index=row['image_index']
    assert name==f'image{index:03d}.json' and 0<=index<100
    image=plan['images'][index];assert sha(r/image['path'])==row['source_image_sha256']==image['sha256']
    assert row['plan_sha256']==progress['plan_sha256'] and row['checkpoint_sha256']==plan['checkpoint_sha256']
    assert sha(path.with_suffix('.ch65'))==row['container_sha256'] and sha(path.with_suffix('.npz'))==row['reconstruction_sha256']
    blob=path.with_suffix('.ch65').read_bytes();parsed=unpack(blob);packets,budget=packetize(blob)
    assert reassemble(packets,len(blob))==blob and budget==row['packet_budget']==packet_budget(len(blob))
    assert parsed['model']=='cheng2020-anchor' and parsed['quality']==1 and parsed['height']==parsed['width']==256
    assert sum(len(s) for s in parsed['streams'])*8/65536==row['entropy_payload_bpp'] and len(blob)*8/65536==row['total_stored_bpp']
    with np.load(path.with_suffix('.npz')) as saved:pixels=saved['pixels']
    assert pixels.dtype==np.float32 and pixels.shape==(1,3,256,256) and np.isfinite(pixels).all() and pixels.min()>=0 and pixels.max()<=1
    assert hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest()==row['decoded_pixels_sha256']
    with Image.open(r/image['path']) as opened:original=np.asarray(opened.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
    score=-10*math.log10(float(np.mean(np.square(pixels-original),dtype=np.float64)));assert abs(score-row['psnr_decoded'])<=1e-4
    assert all(math.isfinite(row[k]) for k in ('estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped'))
    assert row['lpips_normalized']>=0 and -1.00001<=row['clip_clipped']<=1.00001 and not row['channel_simulated'] and not row['paper_baseline_ready']
    if model is not None and index in plan['fresh_decoder_image_indices']:
        with torch.inference_mode():decoded=model.decompress([[s] for s in parsed['streams']],parsed['shape'])['x_hat'].cpu().numpy()
        error=float(np.max(np.abs(decoded-pixels)));assert error==0
        fresh.append(dict(image_index=index,max_abs_pixel_difference=error,encoder_called=False))
    rows.append(row)
assert len(rows)==progress['completed_cases'] and len({v['image_index'] for v in rows})==len(rows)
assert set(progress['case_sha256'])=={f.name for f in target.glob('image[0-9][0-9][0-9].json')}
complete=len(rows)==100 and len(fresh)==5
means=None
if complete:
    fields=('entropy_payload_bpp','total_stored_bpp','estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped')
    means={k:math.fsum(v[k] for v in rows)/100 for k in fields}
    means.update({k:math.fsum(v['packet_budget'][k] for v in rows)/100 for k in ('packets','tail_padding_bits','forward_wire_bits','forward_qam_symbols')})
audit=dict(all_checks_passed=True,complete=complete,completed_cases=len(rows),expected_cases=100,
    script_sha256=sha(__file__),plan_sha256=progress['plan_sha256'],progress_sha256=sha(run/'stage66_holdout_progress.json'),
    checkpoint_sha256=plan['checkpoint_sha256'],fresh_decoder_checks=fresh,means=means,case_sha256=progress['case_sha256'],
    historical_artifacts_unchanged=len(plan['historical_evidence_sha256']),no_training=True,no_checkpoint_reselection=True,
    channel_simulated=False,paper_baseline_ready=False,paper_dataset_identity_established=False,full_paper_reproduction_complete=False)
(run/'stage66_holdout_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in audit.items() if k!='case_sha256'},indent=2))
