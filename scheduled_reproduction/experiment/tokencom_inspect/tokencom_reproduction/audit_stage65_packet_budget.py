import json,hashlib
from pathlib import Path
from cheng_packet_budget import packetize,reassemble
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((run/'stage65_cheng_audit.json').read_text(encoding='utf-8'));assert a['all_checks_passed']
rows=[]
for name,digest in a['case_sha256'].items():
    p=run/'stage65_cheng_cases'/name;assert sha(p)==digest
    row=json.loads(p.read_text(encoding='utf-8'));data=p.with_suffix('.ch65').read_bytes();assert hashlib.sha256(data).hexdigest()==row['container_sha256']
    packets,budget=packetize(data);assert reassemble(packets,len(data))==data
    assert budget['entropy_payload_bytes']*8==row['entropy_payload_bits']
    rows.append(dict(model=row['model'],image_index=row['image_index'],**budget))
groups={}
for model in ('cheng2020-anchor','cheng2020-attn'):
    subset=[v for v in rows if v['model']==model]
    if len(subset)==10:groups[model]={k:sum(v[k] for v in subset)/10 for k in ('packets','forward_qam_symbols','forward_wire_bits','tail_padding_bits')}
out=dict(all_checks_passed=True,script_sha256=sha(__file__),helper_sha256=sha(r/'cheng_packet_budget.py'),
    source_audit_sha256=sha(run/'stage65_cheng_audit.json'),cases=len(rows),rows=rows,complete_model_groups=groups,
    scope='Descriptive first-pass integer ledger for custom CH65 containers, not a channel or paper experiment',
    channel_simulated=False,feedback_header_cmi_protocol_complete=False,full_paper_reproduction_complete=False)
(run/'stage65_cheng_packet_budget.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
