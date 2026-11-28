"""Aggregate gate diagnostics by image and package all fixed-threshold outcomes."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'runs'
outputs=root.parents[2]/'outputs'
gate=json.loads((run/'stage40_confidence_gate.json').read_text(encoding='utf-8'))
image_metrics=json.loads((run/'stage40_gate_image_metrics.json').read_text(encoding='utf-8'))['rows']
assert len(gate['rows'])==50 and len(image_metrics)==150
summaries=[]
for threshold in gate['thresholds']:
    policies=[next(p for p in r['policies'] if p['threshold']==threshold) for r in gate['rows']]
    metrics=[r for r in image_metrics if r['threshold']==threshold]
    per_image=[]
    for image in range(10):
        subset=[r for r in metrics if r['image_index']==image]
        assert len(subset)==5
        per_image.append({key:sum(r[key] for r in subset)/5 for key in ['correct_tokens','psnr_raw','lpips_normalized','clip_clipped']})
    for r,p in zip(gate['rows'],policies):
        assert p['correct_tokens']==r['direct_correct']+p['repaired']-p['harmed']
    summaries.append({'threshold':threshold,**{key:sum(p[key] for p in policies) for key in ['replaced','repaired','harmed','correct_tokens']},
                      'image_means':{key:sum(r[key] for r in per_image)/10 for key in per_image[0]},'per_image':per_image})
result={'scope':gate['scope'],'prespecified_thresholds':gate['thresholds'],'threshold_selected':None,'summaries':summaries}
(run/'stage40_gate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
baseline=json.loads((run/'stage37_direct_comparison.json').read_text(encoding='utf-8'))['means']
lines=['# TokCom 第四十阶段：接收端 CRC 置信度门控开发对照','','固定十张开发图、五轮 AWGN 8 dB 接收、真实类别与原作者生成候选。新门控只使用接收 token、CRC、已知类别和模型输出；发送端 token 只在决策后用于计分。使用同一份所有 CRC 失败位置同时遮蔽的上下文，比较候选 token 与接收 token 的 softmax 概率（logits 乘 1.3）。只有候选概率较高且达到预设门槛时才替换。门槛在运行前固定为 0、0.5、0.9，全部报告，未据开发结果选门槛。','','| 路径 | 正确 token /256 | PSNR ↑ | LPIPS ↓ | CLIP ↑ |','|---|---:|---:|---:|---:|']
for label in ['direct','crc']:
    v=baseline[label]
    lines.append(f"| {label} | {v['tokens']:.2f} | {v['psnr_raw']:.4f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
for s in summaries:
    v=s['image_means']
    lines.append(f"| gate {s['threshold']} | {v['correct_tokens']:.2f} | {v['psnr_raw']:.4f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
lines+=['','| 门槛 | 替换数 | 错→对 | 对→错 |','|---|---:|---:|---:|']
for s in summaries:lines.append(f"| {s['threshold']} | {s['replaced']} | {s['repaired']} | {s['harmed']} |")
lines+=['','直接接收正确总数为 12535。三档门控均减少了原 CRC 生成路径的替换和误改，但均未实现精确 token 净修复收益。0.9 门槛的三个替换均为误改，说明在此掩码上下文和候选选择下，模型概率不是已经校准的信道 token 正确率。此结论不能外推至其他门控方法或数据。','','图像指标使用此前固定口径：PSNR 对原图直接比较原始解码输出；LPIPS 对裁剪图 normalize=True；CLIP 使用裁剪后 PIL 和 ViT-B/32。每张图先平均五轮再对十图平均。直接接收、原 CRC 恢复与门控使用相同接收数据；门控额外增加一次模型前向计算，尚未计入时延或资源等价约束。候选生成仍沿用原模型及固定采样种子，不是论文原始 TCE 实现。','','数据仍为开发集十图的重复接收，尚未新盲测；本轮没有训练、校准或选择最终阈值。下一步应在开发集检查 CRC 失败包内的信道软信息能否支持更合适的保留策略，冻结策略后再准备新的留出评估。']
report=outputs/'TokCom_第四十阶段CRC置信度门控对照_2026-11-20.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
names=['evaluate_stage40_confidence_gate.py','prepare_stage40_image_metrics.py','evaluate_stage40_image_metrics.py','report_stage40_gate.py','runs/stage40_confidence_gate.json','runs/stage40_gate_image_metrics.json','runs/stage40_gate_summary.json','runs/stage37_direct_comparison.json']
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage40_crc_confidence_gate.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,base in items:z.write(base/name,name)
    z.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
print(json.dumps(result,indent=2))
print(target)
