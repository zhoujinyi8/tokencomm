"""Save the bounded token-transition diagnosis and its evidence."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
run=root/'runs'
outputs=root.parents[2]/'outputs'
summary=json.loads((run/'stage38_token_transition_summary.json').read_text(encoding='utf-8'))
assert summary['pairs']==10 and all(c['trial']==0 for c in summary['completed_cases'])
link=np.load(run/'stage26_official_token_link_snr8_5trials.npz')
wrong=link['received']!=link['transmitted'][None,:,:]
mask_totals={name:{'masked':int(link[key].sum()),'masked_wrong':int((link[key]&wrong).sum()),
                   'masked_correct':int((link[key]&~wrong).sum())}
             for name,key in [('oracle','oracle_token_mask'),('crc','crc_token_mask')]}
(run/'stage38_all_receptions_mask_counts.json').write_text(json.dumps(mask_totals,indent=2),encoding='utf-8')
report=outputs/'TokCom_第三十八阶段恢复修复与误改来源诊断_2026-11-19.md'
report.write_text('''# TokCom 第三十八阶段：恢复修复与误改来源诊断

本轮保存作者 GenDecoding 最终送入 VQGAN 解码器的 token，记录接收前后每个位置的正确状态。未改变作者生成算法。完成第 0 轮接收的十张开发图；其余四轮的逐位置恢复尚未完成。

| 十图第 0 轮，合计 2560 个位置 | 真值比较遮蔽 | CRC 失败遮蔽 |
|---|---:|---:|
| 接收时错误 token | 59 | 59 |
| 被遮蔽的原本正确 token | 133 | 321 |
| 原本错误，生成后修复 | 4 | 7 |
| 原本正确，生成后变错 | 116 | 288 |
| 被遮蔽且原本正确，生成后仍正确 | 17 | 33 |
| 直接接收正确总数 | 2501 | 2501 |
| 生成后正确总数 | 2389 | 2220 |

净变化精确满足：最终正确数 = 接收正确数 + 修复数 − 误改数。所有最终 token 的未遮蔽位置与接收值相同；重跑 token 正确数及干净 VQGAN MSE 与前阶段记录一致。逐位置最终 token ID 已保存在各 JSON 中。

全部五轮共 50 组的接收缓存有 265 个错误 token。真值比较包级掩码遮蔽 860 个位置，其中 595 个原本正确；CRC 包级掩码遮蔽 1808 个位置，其中 1543 个原本正确。两种掩码在这批缓存中都覆盖了 265 个错误位置。这里的全五轮遮蔽统计与上表单轮生成转移统计覆盖范围不同。

这些证据表明，第 0 轮精确 token 正确数下降的直接来源是误改数超过修复数。包级遮蔽重写了原本正确的位置，其中 CRC 掩码额外重写了更多正确位置。该诊断不能说明逐 token 真值定位可在接收端实现，也不能从 token 净下降直接推断所有图像指标下降；前阶段已观察到 PSNR、LPIPS、CLIP 与 token 指标可能不一致。

数据仅限十张开发图、固定生成种子和 8 dB AWGN；下一步补全其余四轮的逐位置转移，再设计保留接收值或按置信度替换的开发对照，明确其所需接收信息。原始作者仓库保持只读。
''',encoding='utf-8')
names=['prepare_stage38_token_transitions.py','batch_stage38_token_transitions.py',
       'summarize_stage38_token_transitions.py','runs/stage38_token_transition_summary.json',
       'runs/stage38_all_receptions_mask_counts.json','runs/stage26_official_token_link_snr8_5trials.npz']
names += [f'runs/stage38_token_transitions_trial0_image{i}.json' for i in range(10)]
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage38_token_transition_diagnosis.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,base in items:z.write(base/name,name)
    z.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
print(report)
print(target,target.stat().st_size)
