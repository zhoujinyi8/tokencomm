"""Pair direct decoding with the two reconstruction paths and save the report."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root/'runs'
direct = json.loads((run/'stage37_direct_baseline.json').read_text(encoding='utf-8'))['rows']
joint = json.loads((run/'stage36_joint_metrics.json').read_text(encoding='utf-8'))['paired_records']
metrics = ['tokens', 'psnr_raw', 'lpips_normalized', 'clip_clipped']
paired = []
for row in direct:
    prior = next(r for r in joint if (r['trial'],r['image_index']) == (row['trial'],row['image_index']))
    assert row['class_name'] == prior['class_name']
    paired.append({**prior, 'direct': {'tokens': row['correct_tokens'], **{m:row[m] for m in metrics[1:]}}})
assert len(paired) == 50
per_image = []
for i in range(10):
    subset = [r for r in paired if r['image_index']==i]
    assert len(subset)==5
    per_image.append({'image_index':i,'class_name':subset[0]['class_name'],
                      **{path:{m:sum(r[path][m] for r in subset)/5 for m in metrics}
                         for path in ('direct','oracle','crc')}})
means = {path:{m:sum(r[path][m] for r in per_image)/10 for m in metrics}
         for path in ('direct','oracle','crc')}
counts = {path:{m:sum((r[path][m]<r['direct'][m] if m=='lpips_normalized' else r[path][m]>r['direct'][m]) for r in per_image)
                for m in metrics} for path in ('oracle','crc')}
result={'images':10,'pairs':50,'means':means,'images_better_than_direct':counts,'per_image':per_image}
(run/'stage37_direct_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2))
lines=['# TokCom 第三十七阶段：十图五轮直接解码基线','','固定第 26 阶段接收 token，直接通过作者 VQGAN 解码，不做生成恢复。与前阶段相同原图、PSNR 原始输出口径、裁剪后归一化 LPIPS 和裁剪后 CLIP 配对。每张图先平均五轮，再对十张图求均值。','','| 路径 | 正确 token /256 | PSNR dB | LPIPS ↓ | CLIP ↑ |','|---|---:|---:|---:|---:|']
for path in means:
    v=means[path]
    lines.append(f"| {path} | {v['tokens']:.2f} | {v['psnr_raw']:.4f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
lines+=['','逐图五轮均值中，优于直接解码的图像数：','',json.dumps(counts,ensure_ascii=False,indent=2),'','direct 为直接接收解码，oracle 为发送端真值比较遮蔽，crc 为接收端 CRC 失败遮蔽。三条路径使用相同接收数据。两种恢复方式掌握的信息不同；开发集十图不能用于论文总体性能推断，五轮复用同一图像。正确 token 数包含所有位置；其下降可来自恢复替换了原本已正确接收的 token。直接解码与恢复路径的指标输入定义一致，数值仅属于此固定开发协议。','','机器结果见 runs/stage37_direct_baseline.json 与 runs/stage37_direct_comparison.json。']
(root.parents[2]/'outputs/TokCom_第三十七阶段直接接收解码基线_2026-11-19.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
