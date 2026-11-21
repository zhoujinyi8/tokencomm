"""Report frozen holdout execution coverage without interim performance selection."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from pretrained_vq import sha256

root=Path(__file__).resolve().parent
run=root/'runs';outputs=root.parents[2]/'outputs'
audit=json.loads((run/'stage45_cache_audit.json').read_text(encoding='utf-8'))
completed=[]
for path in sorted((run/'stage46_holdout_recovery').glob('image[0-9][0-9][0-9]_trial[0-4].npz')):
    image=int(path.stem[5:8]);trial=int(path.stem[-1])
    with np.load(path) as x:
        assert all(x[name].shape==(256,) for name in ('direct','oracle','crc','soft_gate_0.05','soft_gate_0.1','soft_gate_0.25'))
        assert x['recovery_lock_sha256'].item()==sha256(run/'stage46_recovery_lock.json')
        phy=run/f'stage45_holdout_phy/image{image:03d}_trial{trial}.npz'
        assert x['source_phy_sha256'].item()==sha256(phy)
    completed.append({'image_index':image,'trial':trial,'sha256':sha256(path)})
progress={'expected_images':100,'expected_receptions':500,'completed_recovery_cases':len(completed),
          'complete_five_trial_images':sum(sum(c['image_index']==i for c in completed)==5 for i in range(100)),
          'complete':len(completed)==500,'metrics_aggregated':False,'cases':completed}
(run/'stage46_recovery_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
report=outputs/'TokCom_第四十五至四十六阶段新留出执行进度_2026-11-22.md'
report.write_text(f'''# TokCom 第四十五至四十六阶段：新留出执行进度

按照第四十四阶段冻结协议，已经完成作者 token 编码 {audit['encoded_images']}/100 张、物理接收缓存 {audit['phy_cases']}/500 组。每个图像 token 文件和每次接收文件均原子保存，并检查图像身份、形状、取值范围、种子和协议哈希。新留出集现在已经进入固定评估流程，不能再作为策略调参数据。

作者恢复完成 {progress['completed_recovery_cases']}/500 组，其中五轮齐全的图像 {progress['complete_five_trial_images']}/100。每组保存直接接收 token、作者真值比较掩码恢复、接收端 CRC 掩码恢复，以及 5%、10%、25% 软路径代价门控输出。没有读取新留出指标来选择分支、比例或模型。若不完整，应续跑缺项，不把已完成部分当作全部 100 图的最终性能。

编码、物理层和恢复入口分别有执行前脚本哈希锁，续跑发现脚本或协议不一致会拒绝继续。作者参数、真实类别、随机种子、掩码、分包和门控定义沿用冻结协议。完整的恢复 PSNR、LPIPS、CLIP 汇总尚未计算。

GitHub 同步目标为已有项目 zhoujinyi8/tokencomm 的独立分支 reproduction/automation-30-20261122，内容放在 scheduled_reproduction，保持与仓库原有阶段编号的区别。同步代码、报告、冻结清单和机器审计，不上传留出原图、预训练权重或大体积生成缓存。实际推送结果以本轮回复提供的提交链接为准。

下一步：继续完成缺失的恢复组，锁定并运行图像指标入口，再按每张图五轮均值进行全 100 图的配对汇总。当前仍是单一 8 dB、固定生成种子和作者类别信息假设，未实现论文完整 TCE/反馈协议；项目级数据去重不能排除公开预训练重叠。
''',encoding='utf-8')
names=['cache_stage45_holdout_tokens.py','cache_stage45_holdout_phy.py','audit_stage45_caches.py',
       'audit_stage46_outputs.py','record_stage46_provenance.py',
       'evaluate_stage47_frozen_metrics.py','summarize_stage47_complete_holdout.py','restore_stage44_images.py',
       'run_stage46_frozen_recovery.py','runs/stage45_encoder_lock.json','runs/stage45_phy_lock.json',
       'runs/stage45_cache_audit.json','runs/stage46_recovery_progress.json','runs/stage44_frozen_protocol.json']
if (run/'stage46_recovery_lock.json').exists():names.append('runs/stage46_recovery_lock.json')
for name in ['stage46_output_audit.json','stage46_environment.json']:
    if (run/name).exists():names.append('runs/'+name)
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage45_46_frozen_holdout_progress.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,base in items:z.write(base/name,name)
    z.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in manifest.items())
print(json.dumps({k:v for k,v in progress.items() if k!='cases'},indent=2))
print(report)
