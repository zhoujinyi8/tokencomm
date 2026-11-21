"""Extend the TokCom progress overview with validated stage17 metrics."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root.parents[2] / 'outputs'
old = out / 'TokCom_复现总览_截至2026-10-29.md'
new = out / 'TokCom_复现总览_截至2026-11-01.md'
if new.exists(): raise ValueError('New overview already exists')
result = json.loads((root / 'runs/stage17_resnet_feature_full/analysis.json').read_text(encoding='utf-8'))
assert result['images'] == 100
cos = result['mean_cosine_to_original']
row = (f"| 第十七阶段 | 同一 100 图开发验证集，冻结 ResNet18 图像特征代理指标 | "
       f"预训练恢复与原图平均余弦相似度 **{cos['pretrained_masked_class']:.4f}**，空间填补 "
       f"**{cos['spatial']:.4f}** | 不是论文 CLIP/LPIPS；ImageNet 预训练重叠风险，且未做新盲测 |")
content = old.read_text(encoding='utf-8')
content = content.replace('截至 2026-10-29', '截至 2026-11-01')
content = content.replace('接收端恢复、图像重建、重传选包和反馈控制包。',
                          '接收端恢复、图像重建、重传选包、反馈控制包及图像特征诊断。')
anchor = '| 第十六阶段 |'
lines = content.splitlines()
index = next(i for i, line in enumerate(lines) if line.startswith(anchor))
lines.insert(index + 1, row)
content = '\n'.join(lines) + '\n'
content = content.replace('第十一至十六阶段', '第十一至十七阶段')
content = content.replace('第十至十六阶段', '第十至十七阶段')
content = content.replace('2. 增加语义或感知质量指标；保留 PSNR 和精确 token 命中率作为不同层面的诊断。',
                          '2. 继续补齐论文 CLIP/LPIPS 或等价语义与感知评测；保留 ResNet18 代理指标、PSNR 与 token 命中率的不同边界。')
new.write_text(content, encoding='utf-8')
print(new)
