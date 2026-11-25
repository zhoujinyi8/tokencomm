"""Reuse the fixed direct-decoding metrics for all prespecified gate outputs."""
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'evaluate_stage37_direct_baseline.py').read_text(encoding='utf-8')
source=source.replace('rows = []', "gate=json.loads((run/'stage40_confidence_gate.json').read_text(encoding='utf-8'))\ngate_index={(r['trial'],r['image_index']):r for r in gate['rows']}\nrows = []")
start=source.index('    for image in range(10):')
end=source.index('\nresult =',start)
block=source[start:end]
block=block.replace("tokens = torch.from_numpy(link['received'][trial, image].copy()).long().reshape(1, 16, 16)", "policy=next(p for p in gate_index[(trial,image)]['policies'] if p['threshold']==threshold)\n            tokens=torch.tensor(policy['final_token_ids']).long().reshape(1,16,16)")
block=block.replace("'trial': trial,", "'threshold':threshold, 'trial': trial,")
block=block.replace("int((link['received'][trial, image] == link['transmitted'][image]).sum())", "int((tokens.numpy().reshape(-1) == link['transmitted'][image]).sum())")
source=source[:start]+"    for threshold in (0.,0.5,0.9):\n"+'\n'.join('    '+line for line in block.splitlines())+source[end:]
source=source.replace('stage37_direct_baseline.json','stage40_gate_image_metrics.json')
source=source.replace('direct received-token decoding','prespecified receiver CRC confidence-gate outputs')
(root/'evaluate_stage40_image_metrics.py').write_text(source,encoding='utf-8')
