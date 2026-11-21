"""Cache frozen holdout receptions and coded LLRs with atomic per-case writes."""
import json
import os
import time
import hashlib
from pathlib import Path
import numpy as np
from smoke_stage25_sionna_link import simulate

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

root=Path(__file__).resolve().parent
run=root/'runs'
protocol_path=run/'stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
folder=root/'data/imagenette_stage44_holdout'
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(folder/'test.jsonl')==manifest['test_jsonl_sha256']
assert sha256(root/'smoke_stage25_sionna_link.py')==protocol['script_sha256']['smoke_stage25_sionna_link.py']
records=[json.loads(s) for s in (folder/'test.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records)==100
identity={'protocol_sha256':sha256(protocol_path),'phy_script_sha256':sha256(Path(__file__)),
          'encoder_lock_sha256':sha256(run/'stage45_encoder_lock.json')}
lock=run/'stage45_phy_lock.json'
if lock.exists():assert json.loads(lock.read_text(encoding='utf-8'))==identity
else:lock.write_text(json.dumps(identity,indent=2),encoding='utf-8')
target=run/'stage45_holdout_phy';target.mkdir(exist_ok=True)
deadline=time.monotonic()+45*60
for image,record in enumerate(records):
    with np.load(run/f'stage45_holdout_tokens/image{image:03d}.npz') as saved:
        assert saved['source_sha256'].item()==record['sha256']
        tokens=saved['tokens'].copy()
    for trial in range(5):
        out=target/f'image{image:03d}_trial{trial}.npz'
        if out.exists():
            with np.load(out) as saved:assert saved['source_sha256'].item()==record['sha256']
            continue
        if time.monotonic()>deadline:raise SystemExit('TIME_BUDGET_REACHED; resume remaining cases')
        seed=20261121+100*image+trial
        result,rx,oracle,crc,soft=simulate(seed,8.,tokens=tokens,return_soft=True)
        temporary=out.with_suffix('.partial.npz')
        np.savez_compressed(temporary,received=rx.reshape(256),oracle_token_mask=np.tile(oracle,4),
                            crc_token_mask=np.tile(crc,4),coded_llr=soft['coded_llr'],
                            reencoded_bits=soft['reencoded_decoded_bits'],
                            source_sha256=record['sha256'],metadata=json.dumps(result))
        os.replace(temporary,out)
    print(f'cached image={image+1}/100',flush=True)
print('complete; no aggregate recovery metrics computed',flush=True)
