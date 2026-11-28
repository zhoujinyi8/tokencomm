"""Complete CRC endpoint quality with 299 byte-identical references plus one measured corruption."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
base=read('stage66_transport_endpoints.json');measured=read('stage66_crc_miss_metrics.json');plan=read('stage66_crc_miss_probe_plan.json');lock=read('stage66_crc_miss_metrics_lock_v2.json')
assert base['all_checks_passed'] and measured['all_checks_passed'] and measured['lock_sha256']==sha(run/'stage66_crc_miss_metrics_lock_v2.json')
assert lock['script_sha256']==sha(r/'evaluate_stage66_crc_miss_metrics.py') and lock['probe_result_sha256']==sha(run/'stage66_crc_miss_probe.json')
assert lock['metric_lock_sha256']==sha(run/'stage47_metrics_lock.json')
with np.load(run/'stage66_crc_miss_probe/metrics_pixels.npz') as a:pixels=a['pixels']
assert sha(run/'stage66_crc_miss_probe/metrics_pixels.npz')==measured['reconstruction_sha256']
assert pixels.shape==(1,3,256,256) and np.isfinite(pixels).all() and hashlib.sha256(pixels.astype('<f4',copy=False).tobytes()).hexdigest()==measured['pixel_sha256']
with Image.open(r/plan['original_image_path']) as source:original=np.asarray(source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)[None]/np.float32(255)
assert abs(-10*math.log10(float(np.mean(np.square(pixels-original),dtype=np.float64)))-measured['psnr_decoded'])<1e-4
rows=[]
for v in base['rows']:
    if v['policy']!='receiver_crc_arq_padded':continue
    row=dict(v)
    if row['metrics'] is None:
        assert all(row[k]==measured[k] for k in ('image_index','trial','snr_db'))
        row.update(endpoint='actual_corrupt_stream_decode',metrics={k:measured[k] for k in ('psnr_decoded','lpips_normalized','clip_clipped')},new_decoder_forwards=1)
    rows.append(row)
assert len(rows)==300 and sum(v['new_decoder_forwards'] for v in rows)==1
groups={}
for snr in (6.,6.5,7.,7.5,8.,9.):
    subset=[v for v in rows if v['snr_db']==snr];assert len(subset)==50 and all(v['metrics'] is not None for v in subset)
    groups[str(snr)]=dict(frames=50,exact_clean_reference_rows=sum(v['endpoint']=='exact_byte_clean_reference' for v in subset),
        actual_corrupt_decoder_rows=sum(v['endpoint']=='actual_corrupt_stream_decode' for v in subset),
        means={k:math.fsum(v['metrics'][k] for v in subset)/50 for k in ('psnr_decoded','lpips_normalized','clip_clipped')})
value=dict(all_checks_passed=True,script_sha256=sha(__file__),source_endpoint_rows_sha256=sha(run/'stage66_transport_endpoints.json'),
    measured_corrupt_metric_sha256=sha(run/'stage66_crc_miss_metrics.json'),complete_crc_endpoint_frames=300,clean_reference_rows=299,
    actual_corrupt_decoder_rows=1,new_decoder_forwards_in_completion=1,no_failure_imputation=True,per_snr=groups,
    noisy_single_pass_quality_complete=False,interpretation='Engineering CRC-ARQ endpoints on ten development images; all 300 outcomes retained, 299 exact-byte references and one actual corrupted-stream metric. Not paper-wide baseline reproduction.',
    paper_baseline_ready=False,full_paper_reproduction_complete=False)
(run/'stage66_crc_endpoint_completion.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value,indent=2))
