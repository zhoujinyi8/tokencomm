"""Evaluate the saved per-reception soft-channel gate outputs with fixed metrics."""
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'evaluate_stage40_image_metrics.py').read_text(encoding='utf-8')
source=source.replace('stage40_confidence_gate.json','stage42_soft_gate.json')
source=source.replace("gate_index={(r['trial'],r['image_index']):r for r in gate['rows']}","gate_index={(r['trial'],r['image_index'],r['fraction']):r for r in gate['rows']}")
source=source.replace('for threshold in (0.,0.5,0.9):','for threshold in (.05,.1,.25):')
source=source.replace("policy=next(p for p in gate_index[(trial,image)]['policies'] if p['threshold']==threshold)","policy=gate_index[(trial,image,threshold)]")
source=source.replace('stage40_gate_image_metrics.json','stage43_soft_gate_image_metrics.json')
source=source.replace('prespecified receiver CRC confidence-gate outputs','per-reception soft-channel risk fractions; development-only')
(root/'evaluate_stage43_image_metrics.py').write_text(source,encoding='utf-8')
