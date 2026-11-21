"""Freeze the next protocol and select project-disjoint images without inference."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from prepare_stage7_holdout import sample
from prepare_stage10_holdout import prior_hashes
from prepare_stage5_imagenette import pixel_digest
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
source=Path('D:/新建文件夹 (2)/2026-09-21/ban/work/downloads/imagenette_train.parquet')
target=root/'data/imagenette_stage44_holdout'
protocol_path=root/'runs/stage44_frozen_protocol.json'
if protocol_path.exists() or target.exists():
    raise ValueError('Protocol or dataset already exists; do not overwrite a frozen selection')
prior=[p for p in (root/'data').iterdir() if p.is_dir() and any((p/f'{s}.jsonl').exists() for s in ('train','val','test'))]
file_hashes,pixel_hashes=prior_hashes(prior)
# Prior public sample images may not appear in split manifests.
public=root/'data/public_samples'
public_count=0
for path in public.rglob('*'):
    if path.suffix.lower() in ('.jpg','.jpeg','.png'):
        raw=path.read_bytes()
        file_hashes.add(hashlib.sha256(raw).hexdigest())
        pixel_hashes.add(pixel_digest(raw));public_count+=1
scripts=['smoke_stage25_sionna_link.py','batch_stage33_author_psnr.py',
         'evaluate_stage37_direct_baseline.py','evaluate_stage42_soft_gate.py',
         'evaluate_stage43_image_metrics.py','prepare_stage44_holdout.py']
weights=['pretrained/official_tokcom/last.ckpt','pretrained/official_tokcom/MaskGIT_ImageNet_256.pth']
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),
          'status':'frozen_before_selection_and_inference',
          'purpose':'author-path reproduction on new project-disjoint data; no further tuning on holdout',
          'images':100,'per_class':10,'selection_seed':20261121,
          'source_kind':'unused images from local Imagenette training-source parquet; not upstream test split',
          'source_sha256':sha256(source),'snr_db':[8.0],'phy_trials':5,
          'phy_seed_formula':'20261121 + 100 * image_index + trial',
          'generation_seed':20261121,'true_imagenet_class_available':True,
          'preprocessing':'RGB Resize((256,256)) + ToTensor; no crop or further normalization for author VQGAN',
          'author_generation':{'step':32,'sm_temp':1.3,'w':1,'randomize':'linear','r_temp':7,'sched_mode':'arccos'},
          'paths':['direct_received_decode','author_oracle_packet_mask','receiver_crc_packet_mask'],
          'exploratory_paths':{'soft_path_cost_gate_fractions':[.05,.1,.25],
                               'candidate':'saved receiver CRC full-mask generation',
                               'selection':'per-reception CRC-mask ranking; stable sort; ceil fraction; no chosen fraction'},
          'metrics':{'token':'all 256 exact IDs','psnr':'raw decoded output vs original, data_range=1',
                     'lpips':'0.1.4 AlexNet; clamp decoded output [0,1]; normalize=True',
                     'clip':'ViT-B/32; clamp decoded output [0,1]; ToPILImage; official preprocess; feature cosine'},
          'aggregation':'five trials averaged within each image, then average 100 images; paired per-image differences',
          'reporting':'all paths and all metrics; exploratory gates not selected by holdout outcome; incomplete cases reported as incomplete',
          'primary_descriptive_comparison':'author oracle-mask reconstruction vs direct decoding, with explicit extra-information caveat',
          'limits':['project-level file and decoded-pixel disjointness only; upstream ImageNet pretraining overlap not excluded',
                    'single SNR, fixed generation seed; no full TCE or feedback protocol reproduction'],
          'script_sha256':{name:sha256(root/name) for name in scripts},
          'weight_sha256':{name:sha256(root/name) for name in weights},
          'excluded_datasets':[str(p.relative_to(root)) for p in prior],
          'excluded_file_hashes':len(file_hashes),'excluded_pixel_hashes':len(pixel_hashes),
          'public_sample_images_checked':public_count}
protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
selected,names,eligible,rejected=sample(source,10,20261121,file_hashes,pixel_hashes)
records=[];seen_files=set();seen_pixels=set()
for label,group in enumerate(selected):
    folder=target/'images/test'/str(label);folder.mkdir(parents=True,exist_ok=False)
    for raw,digest,pixels,source_path in group:
        assert digest not in seen_files|file_hashes and pixels not in seen_pixels|pixel_hashes
        seen_files.add(digest);seen_pixels.add(pixels)
        path=folder/(digest[:20]+'.jpg');path.write_bytes(raw)
        records.append({'image':path.relative_to(target).as_posix(),'label':label,
                        'answer':names[label].lower(),'sha256':digest,'pixel_sha256':pixels,'source_path':source_path})
listing=target/'test.jsonl'
listing.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
manifest={'count':len(records),'class_names':names,'per_class':10,'eligible_by_class':eligible,
          'rejected_prior_rows':rejected,'source_sha256':protocol['source_sha256'],
          'test_jsonl_sha256':sha256(listing),'protocol_sha256':sha256(protocol_path),
          'prior_file_overlap':0,'prior_pixel_overlap':0,'inference_status':'not_encoded_or_evaluated',
          'note':'pixel hashes computed for deduplication only; no model inference or visual selection'}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
