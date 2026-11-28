"""Audit actual clean-codec bytes and pixels without relabeling stock weights as paper weights."""
import hashlib,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image
from cheng_stream_container import unpack
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
plan=read('stage65_cheng_diagnostic_plan.json');lock=read('stage65_cheng_diagnostic_lock.json');progress=read('stage65_cheng_progress.json')
assert lock['script_sha256']==sha(r/'run_stage65_cheng_diagnostic.py') and lock['auditor_sha256']==sha(__file__)
assert lock['fresh_decoder_sha256']==sha(r/'decode_cheng_container.py')
assert lock['plan_sha256']==progress['plan_sha256']==sha(run/'stage65_cheng_diagnostic_plan.json')
assert progress['lock_sha256']==sha(run/'stage65_cheng_diagnostic_lock.json')
assert lock['container_helper_sha256']==plan['container_helper_sha256']==sha(r/'cheng_stream_container.py')
assert plan['container_test_sha256']==sha(r/'test_cheng_stream_container.py')
assert plan['metric_lock_sha256']==sha(run/'stage47_metrics_lock.json')
cases=[];hashes={};coordinates=set();fresh_checks=[]
target=run/'stage65_cheng_cases'
for p in sorted(target.glob('model[01]_image[0-9][0-9].json')):
    mi,i=map(int,re.fullmatch(r'model([01])_image([0-9]{2})',p.stem).groups());assert i<10 and (mi,i) not in coordinates
    coordinates.add((mi,i));row=json.loads(p.read_text(encoding='utf-8'));assert sha(p)==progress['case_sha256'][p.name]
    assert row['model']==plan['models'][mi] and row['image_index']==i and row['quality']==1 and row['paper_baseline_ready'] is False
    assert row['lock_sha256']==sha(run/'stage65_cheng_diagnostic_lock.json') and row['weight_sha256']==plan['weights'][mi]['sha256']
    image=plan['images'][i];assert sha(r/image['path'])==row['source_image_sha256']==image['sha256']
    container=p.with_suffix('.ch65');array=p.with_suffix('.npz')
    assert sha(container)==row['container_sha256'] and sha(array)==row['reconstruction_sha256']
    parsed=unpack(container.read_bytes());assert parsed['model']==row['model'] and parsed['quality']==1 and parsed['height']==parsed['width']==256
    assert list(parsed['shape'])==row['shape'] and [len(s) for s in parsed['streams']]==row['entropy_stream_bytes']
    bits=sum(row['entropy_stream_bytes'])*8;header=parsed['header_bytes']*8;total=container.stat().st_size*8
    assert header==272 and bits==row['entropy_payload_bits'] and header==row['container_header_bits'] and total==row['total_stored_bits']==bits+header
    assert row['entropy_payload_bpp']==bits/65536 and row['total_stored_bpp']==total/65536
    assert all(math.isfinite(row[k]) for k in ('estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped','encode_decode_seconds'))
    with np.load(array) as saved:pixels=saved['pixels']
    assert pixels.shape==(1,3,256,256) and pixels.dtype==np.float32 and np.isfinite(pixels).all() and pixels.min()>=0 and pixels.max()<=1
    assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['decoded_pixels_sha256']
    with Image.open(r/image['path']) as source:
        original=np.asarray(source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
    mse=float(np.mean(np.square(pixels-original),dtype=np.float64));assert math.isclose(-10*math.log10(mse),row['psnr_decoded'],rel_tol=0,abs_tol=1e-4)
    cases.append(row);hashes[p.name]=sha(p)
assert len(cases)==progress['completed_cases'] and set(hashes)==set(progress['case_sha256'])
for mi in range(2):
    fp=run/f'stage65_fresh_model{mi}.json'
    if not fp.exists():continue
    v=json.loads(fp.read_text(encoding='utf-8'));base=target/f'model{mi}_image00.json';row=json.loads(base.read_text(encoding='utf-8'))
    assert v['model']==row['model'] and v['image_index']==0 and v['container_sha256']==row['container_sha256'] and v['weight_sha256']==row['weight_sha256']
    assert v['decoder_script_sha256']==lock['fresh_decoder_sha256'] and not v['source_image_loaded'] and not v['encoder_called']
    assert sha(run/f'stage65_fresh_model{mi}.npz')==v['output_sha256']
    with np.load(run/f'stage65_fresh_model{mi}.npz') as a,np.load(target/f'model{mi}_image00.npz') as b:
        assert hashlib.sha256(a['pixels'].tobytes()).hexdigest()==v['decoded_pixels_sha256']
        error=float(np.max(np.abs(a['pixels']-b['pixels'])));assert error<=1e-6
    fresh_checks.append(dict(model=v['model'],image_index=0,max_abs_pixel_difference=error,receipt_sha256=sha(fp)))
groups={}
for model in plan['models']:
    subset=[v for v in cases if v['model']==model]
    if len(subset)!=10:continue
    assert {v['image_index'] for v in subset}==set(range(10))
    fields=('entropy_payload_bpp','total_stored_bpp','estimated_likelihood_bpp','psnr_decoded','lpips_normalized','clip_clipped')
    means={k:sum(v[k] for v in subset)/10 for k in fields}
    groups[model]=dict(cases=10,means=means,actual_payload_bpp_range=[min(v['entropy_payload_bpp'] for v in subset),max(v['entropy_payload_bpp'] for v in subset)],
        payload_rate_ratio_to_paper_target=means['entropy_payload_bpp']/plan['paper_target_bpp'],
        payload_rate_within_diagnostic_tolerance=abs(means['entropy_payload_bpp']-plan['paper_target_bpp'])<=plan['paper_target_bpp']*plan['diagnostic_rate_tolerance_relative'],
        paper_baseline_ready=False,reason='Stock public weights; paper fine-tuned checkpoint and exact ImageNet100 identities are not established')
result=dict(all_checks_passed=True,audit_script_sha256=sha(__file__),plan_sha256=sha(run/'stage65_cheng_diagnostic_plan.json'),lock_sha256=sha(run/'stage65_cheng_diagnostic_lock.json'),
    completed_cases=len(cases),expected_cases=20,complete=len(cases)==20 and len(fresh_checks)==2,codec_cases_complete=len(cases)==20,
    fresh_process_checks=fresh_checks,complete_model_groups=groups,case_sha256=hashes,paper_baseline_ready=False,
    training_performed=False,channel_simulated=False,existing_studies_changed=False,full_paper_reproduction_complete=False)
(run/'stage65_cheng_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='case_sha256'},indent=2))
