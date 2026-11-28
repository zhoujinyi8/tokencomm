"""Audit cache identities/completeness without aggregating holdout performance."""
import json
from pathlib import Path
import numpy as np
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs';folder=root/'data/imagenette_stage44_holdout'
protocol_path=run/'stage44_frozen_protocol.json'
records=[json.loads(s) for s in (folder/'test.jsonl').read_text(encoding='utf-8').splitlines()]
token_count=phy_count=0;hashes={}
for image,record in enumerate(records):
    token_path=run/f'stage45_holdout_tokens/image{image:03d}.npz'
    if not token_path.exists():continue
    with np.load(token_path) as x:
        assert x['source_sha256'].item()==record['sha256']
        assert x['tokens'].shape==(256,) and x['tokens'].min()>=0 and x['tokens'].max()<1024
    token_count+=1;hashes[str(token_path.relative_to(root))]=sha256(token_path)
    for trial in range(5):
        path=run/f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz'
        if not path.exists():continue
        with np.load(path) as x:
            assert x['source_sha256'].item()==record['sha256']
            assert x['received'].shape==(256,) and x['received'].min()>=0 and x['received'].max()<1024
            assert x['crc_token_mask'].shape==x['oracle_token_mask'].shape==(256,)
            assert x['coded_llr'].shape==x['reencoded_bits'].shape==(64,102)
            assert np.isfinite(x['coded_llr']).all()
            metadata=json.loads(x['metadata'].item())
            assert metadata['seed']==20261121+100*image+trial and metadata['snr_db']==8.
        phy_count+=1;hashes[str(path.relative_to(root))]=sha256(path)
result={'holdout_images':100,'encoded_images':token_count,'phy_cases':phy_count,'expected_phy_cases':500,
        'complete':token_count==100 and phy_count==500,
        'protocol_sha256':sha256(protocol_path),'recovery_metrics_computed':False,
        'note':'token encoding and channel simulation have now accessed the frozen holdout; no strategy tuning authorized from these outputs',
        'cache_sha256':hashes}
(run/'stage45_cache_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cache_sha256'},indent=2))
