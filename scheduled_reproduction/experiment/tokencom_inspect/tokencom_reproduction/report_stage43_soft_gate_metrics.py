"""Pair soft-gate image metrics with the direct and original CRC baselines."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'runs'
outputs=root.parents[2]/'outputs'
data=json.loads((run/'stage43_soft_gate_image_metrics.json').read_text(encoding='utf-8'))
gate=json.loads((run/'stage42_soft_gate.json').read_text(encoding='utf-8'))
baseline=json.loads((run/'stage37_direct_comparison.json').read_text(encoding='utf-8'))
index={(r['trial'],r['image_index'],r['fraction']):r for r in gate['rows']}
assert len(data['rows'])==150
summaries=[]
keys=['correct_tokens','psnr_raw','lpips_normalized','clip_clipped']
for fraction in (.05,.1,.25):
    rows=[r for r in data['rows'] if r['threshold']==fraction]
    images=[]
    for image in range(10):
        subset=[r for r in rows if r['image_index']==image]
        assert len(subset)==5 and len({r['trial'] for r in subset})==5
        for row in subset:assert row['correct_tokens']==index[(row['trial'],image,fraction)]['correct']
        means={k:sum(r[k] for r in subset)/5 for k in keys}
        images.append({'image_index':image,**means})
    means={k:sum(r[k] for r in images)/10 for k in keys}
    counts={k:sum((r[k]<baseline['per_image'][r['image_index']]['direct'][k] if k=='lpips_normalized' else
                  r[k]>baseline['per_image'][r['image_index']]['direct']['tokens' if k=='correct_tokens' else k])
                 for r in images) for k in keys if k!='correct_tokens'}
    summaries.append({'fraction':fraction,'means':means,'images_better_than_direct':counts,'per_image':images})
result={'scope':data['scope'],'images':10,'trials':5,'selected_fraction':None,'summaries':summaries}
(run/'stage43_soft_gate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# TokCom 第四十三阶段：信道软信息门控图像质量对照','','完成第 42 阶段保存的 5%、10%、25% 三档门控输出，每档十张开发图 × 五轮接收，共 150 次图像解码评估。沿用固定原图与 VQGAN/LPIPS/CLIP 权重。决策先前已经保存，本轮仅解码计分，没有修改比例或生成候选。每图先平均五轮，再对十图平均。','','| 路径 | 正确 token /256 ↑ | PSNR ↑ | LPIPS ↓ | CLIP ↑ |','|---|---:|---:|---:|---:|']
for name in ('direct','crc'):
    v=baseline['means'][name]
    lines.append(f"| {name} | {v['tokens']:.2f} | {v['psnr_raw']:.5f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
for s in summaries:
    v=s['means'];lines.append(f"| soft gate {s['fraction']:.0%} | {v['correct_tokens']:.2f} | {v['psnr_raw']:.5f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
lines+=['','逐图五轮均值优于直接解码的图像数：','',json.dumps({str(s['fraction']):s['images_better_than_direct'] for s in summaries},indent=2),'','PSNR 使用原始解码输出对原图计算；LPIPS 对裁剪图使用 normalize=True；CLIP 对裁剪后 PIL 使用 ViT-B/32。图像文件哈希已核对，正确 token 总数与第 42 阶段保存结果一致。','','direct 为直接解码，crc 为原始 CRC 全掩码生成恢复。软门控沿用该生成候选，每次接收仅把 CRC 遮蔽位置中路径观测代价最高的一定比例替换，其余保持接收值。三档全部报告，未选择最终比例；这是查看开发诊断后提出的对照。信道分数不是解码后 token 的校准后验概率，卷积码跨 token 的状态依赖也没有消失。','','所有结果来自同一批十张开发图、固定生成种子、8 dB AWGN 的重复接收。它们不能视作新留出结果或论文总体性能；均值差不能单独证明总体收益。门控增加信道代理计算，尚未作等算力或时延比较。下一步应收敛开发策略、固定主要评估口径并准备新的留出协议，避免继续以十图开发结果作最终结论。']
report=outputs/'TokCom_第四十三阶段软信息门控图像指标_2026-11-21.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
names=['prepare_stage43_image_metrics.py','evaluate_stage43_image_metrics.py','report_stage43_soft_gate_metrics.py',
       'runs/stage43_soft_gate_image_metrics.json','runs/stage43_soft_gate_summary.json','runs/stage42_soft_gate.json','runs/stage37_direct_comparison.json']
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage43_soft_gate_image_metrics.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,base in items:z.write(base/name,name)
    z.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
print(json.dumps({**result,'summaries':[{k:v for k,v in s.items() if k!='per_image'} for s in summaries]},indent=2))
print(target)
