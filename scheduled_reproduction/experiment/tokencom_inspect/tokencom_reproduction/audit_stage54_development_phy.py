"""Audit planned PHY caches; summarize only complete per-SNR development groups."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from paper_tce_reference import ideal_arq_tce, payload_tce

root=Path(__file__).resolve().parent;run=root/'runs'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
plan_path=run/'stage54_development_multisnr_plan.json'
plan=json.loads(plan_path.read_text(encoding='utf-8'))
progress=json.loads((run/'stage54_development_phy_progress.json').read_text(encoding='utf-8'))
assert progress['plan_sha256']==sha(plan_path)
assert sha(root/'cache_stage54_development_phy.py')==plan['phy_cache_script_sha256']
assert sha(root/'smoke_stage25_sionna_link.py')==plan['phy_simulator_sha256']
with np.load(run/'stage26_official_token_link_snr8_5trials.npz') as link:sent=link['transmitted'].copy()
with np.load(run/'stage26_official_demo_tokens.npz') as source:images=json.loads(source['metadata'].item())['rows']
cases=[];hashes={}
for path in sorted((run/'stage54_development_phy').glob('snr[0-5]_image[0-9][0-9]_trial[0-4].npz')):
    match=re.fullmatch(r'snr([0-5])_image([0-9]{2})_trial([0-4])',path.stem)
    snr_index,image,trial=map(int,match.groups());assert image<10
    assert sha(path)==progress['case_sha256'][path.name]
    with np.load(path) as saved:
        assert saved['plan_sha256'].item()==sha(plan_path)
        assert saved['source_sha256'].item()==images[image]['image_sha256']
        rx=saved['received'];assert rx.shape==(256,) and rx.min()>=0 and rx.max()<1024
        wrong=sent[image]!=rx
        true_bad=wrong.reshape(4,64).any(axis=0)
        oracle=saved['oracle_token_mask'].astype(bool).reshape(4,64)
        crc=saved['crc_token_mask'].astype(bool).reshape(4,64)
        assert np.array_equal(oracle,np.broadcast_to(true_bad,(4,64)))
        assert np.array_equal(crc,np.broadcast_to(crc[0],(4,64)))
        metadata=json.loads(saved['metadata'].item())
        assert metadata['seed']==plan['phy_seed_base']+100*image+trial
        assert metadata['snr_db']==plan['snr_db'][snr_index]
        assert metadata['raw_bits_per_packet']==40 and metadata['crc_bits_per_packet']==51
        assert metadata['coded_bits_per_packet']==102 and metadata['qam_symbols_total']==1632
        assert metadata['token_errors']==int(wrong.sum()) and metadata['oracle_bad_packets']==int(true_bad.sum())
        assert metadata['crc_bad_packets']==int(crc[0].sum())
        bit_errors=int((((sent[image].astype(np.int64)^rx.astype(np.int64))[:,None] & (1<<np.arange(10)))!=0).sum())
        assert metadata['bit_errors']==bit_errors
        cases.append({'snr_index':snr_index,'snr_db':metadata['snr_db'],'image_index':image,'trial':trial,
                      'true_error_packets':int(true_bad.sum()),'crc_failed_packets':int(crc[0].sum()),
                      'crc_failed_data_correct_packets':int((crc[0]&~true_bad).sum()),
                      'crc_passed_data_wrong_packets':int((~crc[0]&true_bad).sum()),
                      'wrong_tokens':int(wrong.sum()),'wrong_payload_bits':bit_errors})
    hashes[path.name]=sha(path)
assert len(cases)==progress['completed_cases']
coverage={};groups={}
for index,snr in enumerate(plan['snr_db']):
    group=[case for case in cases if case['snr_index']==index]
    coverage[str(snr)]=len(group)
    assert len({(c['image_index'],c['trial']) for c in group})==len(group)
    if len(group)==50:
        assert {(c['image_index'],c['trial']) for c in group}=={(i,t) for i in range(10) for t in range(5)}
        per=sum(c['true_error_packets'] for c in group)/(50*64)
        groups[str(snr)]={'complete_receptions':50,'true_packet_error_rate':per,
            'crc_failure_rate':sum(c['crc_failed_packets'] for c in group)/(50*64),
            'received_token_error_rate':sum(c['wrong_tokens'] for c in group)/(50*256),
            'single_shot_payload_tce':payload_tce(),'ideal_arq_payload_tce':ideal_arq_tce(per)}
result={'scope':plan['scope'],'plan_sha256':sha(plan_path),'all_checks_passed':True,
        'completed_cases':len(cases),'expected_cases':300,'complete':len(cases)==300,
        'counts_by_snr':coverage,'complete_snr_groups':groups,'case_sha256':hashes,
        'initial_send_wire_ledger':{'payload_bits':2560,'crc_bits':704,'coded_bits':6528,'qam_symbols':1632,
          'note':'whole-frame mapping; not a single-packet retry or feedback cost'},
        'primary_holdout_changed':False,'cases':cases}
(run/'stage54_development_phy_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','case_sha256')},indent=2))
