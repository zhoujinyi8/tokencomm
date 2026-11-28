"""Cache a preplanned old-development multi-SNR author-demo PHY grid."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from smoke_stage25_sionna_link import simulate

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-minutes',type=float,default=10)
args=parser.parse_args()
assert 0<args.max_minutes<=30
deadline=time.monotonic()+args.max_minutes*60
root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
plan_path=run/'stage54_development_multisnr_plan.json'
plan=json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['phy_cache_script_sha256']==sha(Path(__file__))
assert plan['phy_simulator_sha256']==sha(root/'smoke_stage25_sionna_link.py')
for name,digest in plan['input_sha256'].items():assert sha(root/name)==digest
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as link:sent=link['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as source:rows=json.loads(source['metadata'].item())['rows']
assert sent.shape==(10,256)
target=run/'stage54_development_phy';target.mkdir(exist_ok=True)
budget=False
for image in range(10):
    for trial in range(5):
        for snr_index,snr in enumerate(plan['snr_db']):
            out=target/f'snr{snr_index}_image{image:02d}_trial{trial}.npz'
            if out.exists():
                with np.load(out) as saved:assert saved['plan_sha256'].item()==sha(plan_path)
                continue
            if time.monotonic()>deadline:budget=True;break
            seed=plan['phy_seed_base']+100*image+trial
            metadata,rx,oracle,crc=simulate(seed,float(snr),tokens=sent[image],return_arrays=True)
            assert metadata['raw_bits_per_packet']==40 and metadata['crc_bits_per_packet']==51
            assert metadata['coded_bits_per_packet']==102 and metadata['qam_symbols_total']==1632
            temporary=out.with_suffix('.partial.npz')
            np.savez_compressed(temporary,received=rx.reshape(256),oracle_token_mask=np.tile(oracle,4),
                crc_token_mask=np.tile(crc,4),source_sha256=rows[image]['image_sha256'],
                plan_sha256=sha(plan_path),metadata=json.dumps(metadata))
            os.replace(temporary,out)
            print(f'PHY snr={snr} image={image+1}/10 trial={trial+1}/5',flush=True)
        if budget:break
    if budget:break
files=sorted(target.glob('snr[0-5]_image[0-9][0-9]_trial[0-4].npz'))
counts={str(value):sum(path.name.startswith(f'snr{i}_') for path in files) for i,value in enumerate(plan['snr_db'])}
result={'scope':plan['scope'],'plan_sha256':sha(plan_path),'completed_cases':len(files),'expected_cases':300,
        'complete':len(files)==300,'counts_by_snr':counts,'budget_reached':budget,
        'case_sha256':{path.name:sha(path) for path in files}}
(run/'stage54_development_phy_progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='case_sha256'},indent=2))
