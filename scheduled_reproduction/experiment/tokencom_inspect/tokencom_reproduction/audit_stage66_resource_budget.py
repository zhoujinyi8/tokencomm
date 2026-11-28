"""Byte-exact stream framing sensitivity; fixed first pass only, no feedback or channel inference."""
import hashlib,json,math
from pathlib import Path
from cheng_packet_budget import packet_budget,packetize,reassemble
from cheng_stream_container import unpack
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
grid=read('stage66_rate_grid.json');audit=read('stage66_rate_grid_audit.json')
assert audit['all_checks_passed'] and audit['complete'] and audit['source_result_sha256']==sha(run/'stage66_rate_grid.json')
groups={}
def calculate(cases):
    rows=[]
    for case in cases:
        path=r/case['container_path'];assert sha(path)==case['container_sha256']
        blob=path.read_bytes();parsed=unpack(blob);packets,ledger=packetize(blob)
        assert reassemble(packets,len(blob))==blob
        stream_bytes=sum(map(len,parsed['streams']));assert len(blob)==stream_bytes+34
        assert stream_bytes*8/65536==case['entropy_payload_bpp'] and len(blob)*8/65536==case['total_stored_bpp']
        # Omitting the header is a clearly labeled counterfactual: decoder metadata must arrive separately.
        entropy_only=packet_budget(stream_bytes,header_bytes=0)
        rows.append(dict(image_index=case['image_index'],container_sha256=case['container_sha256'],with_custom_header=ledger,
                         entropy_only_metadata_out_of_band=entropy_only,
                         custom_header_added_qam_symbols=ledger['forward_qam_symbols']-entropy_only['forward_qam_symbols']))
    fields=('packets','tail_padding_bits','crc_bits','convolutional_bits','qam_padding_bits','forward_wire_bits','forward_qam_symbols')
    means={mode:{k:math.fsum(v[mode][k] for v in rows)/len(rows) for k in fields} for mode in ('with_custom_header','entropy_only_metadata_out_of_band')}
    for mode in means:means[mode]['ratio_to_tokcom_1664_symbol_first_pass']=means[mode]['forward_qam_symbols']/1664
    return dict(cases=len(rows),means=means,rows=rows)
for arm in grid['arms']:groups[arm['arm_id']]=calculate(arm['calibration_cases'])
holdout_path=run/'stage66_holdout_audit.json'
if holdout_path.exists():
    held=read(holdout_path.name);assert held['all_checks_passed'] and held['complete']
    cases=[]
    for name,h in held['case_sha256'].items():
        path=run/'stage66_holdout_cases'/name;assert sha(path)==h;v=json.loads(path.read_text(encoding='utf-8'))
        v['container_path']=str(path.with_suffix('.ch65').relative_to(r));cases.append(v)
    groups['frozen_holdout']=calculate(cases)
receipt=dict(all_checks_passed=True,script_sha256=sha(__file__),grid_audit_sha256=sha(run/'stage66_rate_grid_audit.json'),
    holdout_audit_sha256=sha(holdout_path) if holdout_path.exists() else None,groups=groups,
    tokcom_reference=dict(payload_bits=2560,packets=64,forward_qam_symbols=1664,source='completed matched padded framing study; forward data only'),
    metadata_out_of_band_case_is_counterfactual=True,custom_header_is_not_identified_paper_header=True,
    feedback_and_cmi_airtime_included=False,channel_simulated=False,full_resource_equivalence_established=False,paper_baseline_ready=False)
(run/'stage66_resource_budget.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:dict(cases=v['cases'],means=v['means']) for k,v in groups.items()},indent=2))
