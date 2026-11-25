"""Render separate development and ideal-reference reports, preserving primary scope."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'runs';outputs=root.parents[2]/'outputs'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
date=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).date().isoformat()
reports=[]
grid_path=run/'stage53_development_cmi_grid_summary.json'
if grid_path.exists():
    grid=json.loads(grid_path.read_text(encoding='utf-8'))
    audit=json.loads((run/'stage53_development_cmi_grid_audit.json').read_text(encoding='utf-8'))
    assert audit['all_checks_passed'] and audit['summary_sha256']==sha(grid_path)
    text=f'''# TokCom 阶段 53：旧开发集有/无 CMI 对照

本实验使用原有 10 张开发图、五轮接收，完成 {grid['completed_cases']}/50 组；不是当前 100 图冻结留出结果。两个生成路径共用相同接收、oracle 包掩码、生成种子 20261124、最多 32 步及其他作者参数，仅显式切换 w=0/1。没有使用本实验改写主协议或选择门控参数。

全部已保存案例通过计划与源图身份、三条路径、步数、有限值及文件哈希审计；首个案例与预先固定的真实检查点 smoke 一致。指标权重身份与主评估一致。结果见 stage53_development_cmi_grid_summary.json 和对应 audit。
'''
    if grid['complete']:
        text+='\n每张图先平均五轮，再平均 10 张图。以下仅为已观察开发集的描述性结果，不作独立验证或显著性结论。\n\n| 路径 | 正确 token/256 ↑ | PSNR/dB ↑ | LPIPS ↓ | CLIP ↑ |\n|---|---:|---:|---:|---:|\n'
        for name,label in [('direct','直接解码'),('without_cmi','无 CMI'),('with_cmi','有 CMI')]:
            row=grid['means_over_images'][name]
            text+=f"| {label} | {row['correct_tokens']:.4f} | {row['psnr_raw']:.4f} | {row['lpips_normalized']:.6f} | {row['clip_clipped']:.6f} |\n"
        d=grid['difference_with_minus_without_cmi']
        text+=f"\n有 CMI 减无 CMI：正确 token {d['correct_tokens']:+.4f}，PSNR {d['psnr_raw']:+.4f} dB，LPIPS {d['lpips_normalized']:+.6f}，CLIP {d['clip_clipped']:+.6f}。LPIPS 差值为负表示下降。\n"
    else:
        text+='\n预算窗口结束时案例未齐全，没有生成部分图像的性能均值；后续沿相同计划续跑缺项。\n'
    text+='\n当前使用 sender-truth oracle 包掩码和既有开发图，不能据此宣称接收端部署收益或未见数据的 CMI 效果。生成缓存保留本地；计划、脚本、汇总和审计用于复核。\n'
    report=outputs/f'TokCom_阶段53旧开发集CMI对照_{date}.md'
    report.write_text(text,encoding='utf-8');reports.append(report.name)

ideal_path=run/'stage53_ideal_vq_summary.json'
if ideal_path.exists():
    ideal=json.loads(ideal_path.read_text(encoding='utf-8'))
    plan_path=run/'stage53_ideal_vq_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    assert ideal['plan_sha256']==sha(plan_path) and not ideal['physical_retries_simulated']
    files=sorted((run/'stage53_ideal_vq_metrics').glob('image[0-9][0-9][0-9].json'))
    assert len(files)==ideal['completed_images']
    for path in files:
        row=json.loads(path.read_text(encoding='utf-8'))
        assert sha(path)==ideal['case_sha256'][path.name] and row['plan_sha256']==sha(plan_path)
        assert row['correct_tokens']==256 and all(math.isfinite(row[k]) for k in ('psnr_raw','lpips_normalized','clip_clipped'))
    assert ideal['complete']==(len(files)==100)
    if ideal['complete']:
        assert {json.loads(p.read_text(encoding='utf-8'))['image_index'] for p in files}==set(range(100))
    audit={'all_checks_passed':True,'completed_images':len(files),'complete':ideal['complete'],
           'summary_sha256':sha(ideal_path),'source_plan_sha256':sha(plan_path),
           'primary_protocol_unchanged':True,'physical_retries_simulated':False}
    (run/'stage53_ideal_vq_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    text=f'''# TokCom 阶段 53：理想正确 token VQGAN 参考

这是在完整主评估之后预先记录计划的补充参考，不是第 44 阶段六条冻结主路径之一。使用同一 100 张图及已编码的发送端 token，假设接收正确全部 token；每张图只需一次确定性 VQGAN 解码。原有 VQ 量化失真仍存在，正确 token 接收并非像素无损重建，也不构成所有图像指标的数学上界。

完成 {ideal['completed_images']}/100 张。源图、token、VQGAN 和指标权重身份经过核验，PSNR/LPIPS/CLIP 口径与主评估一致。全部保存文件通过哈希、来源计划、正确 token 假设和有限值检查。
'''
    if ideal['complete']:
        row=ideal['means_over_images']
        text+=f"\n| 补充参考 | 正确 token/256 | PSNR/dB ↑ | LPIPS ↓ | CLIP ↑ |\n|---|---:|---:|---:|---:|\n| 理想正确 token VQGAN | {row['correct_tokens']:.4f} | {row['psnr_raw']:.4f} | {row['lpips_normalized']:.6f} | {row['clip_clipped']:.6f} |\n"
        text+=f"\n将该质量参考与此前独立解析核验的理想 ARQ token-payload TCE={ideal['analytic_ideal_arq_payload_tce']:.4f} pixels/token-payload-bit 配对，只形成当前 8 dB 条件下的理想参考点。\n"
    else:
        text+='\n案例未齐全，没有生成部分图像的质量均值。\n'
    text+='\n本实验没有运行实际重传、CRC 停止、ACK 或反馈状态机；解析 TCE 不包括编码、CRC、填充、侧信道及控制/时延成本。不能将它表述为已实测的完整 +R 协议或完整论文曲线。主评估的六条路径和原始汇总保持不变。\n'
    report=outputs/f'TokCom_阶段53理想正确tokenVQGAN参考_{date}.md'
    report.write_text(text,encoding='utf-8');reports.append(report.name)
print(json.dumps({'reports':reports},ensure_ascii=True))
