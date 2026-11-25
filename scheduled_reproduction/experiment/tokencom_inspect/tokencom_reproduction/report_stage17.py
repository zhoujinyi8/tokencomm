"""Write the stage17 development feature report from validated JSON."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out_dir = root.parents[2] / 'outputs'
result = json.loads((root / 'runs/stage17_resnet_feature_full/results.json').read_text(encoding='utf-8'))
analysis = json.loads((root / 'runs/stage17_resnet_feature_full/analysis.json').read_text(encoding='utf-8'))
assert result['images'] == analysis['images'] == 100
cos = analysis['mean_cosine_to_original']
agree = analysis['top1_agreement_to_original']
lo, hi = analysis['paired_image_bootstrap_95pct_difference']
content = f'''# TokCom 论文复现：第十七阶段图像特征相似度开发验证（2026-11-01）

本阶段在 **100 张 stage5 开发验证图**上补充图像表示层面的诊断。图像、PER 30%、每包 4 token、信道种子 `20261021`、预训练 MaskGIT 一步恢复和空间填补，与第十一阶段图像 PSNR 实验相同。每张图的哈希、擦除 token 数及两种方法的 token 命中数均与第十一阶段逐图核对。

使用本地固定权重的 **ImageNet 预训练 ResNet18**。对原图、无丢包 VQGAN 重建、预训练 MaskGIT 恢复图和空间填补图采用同一 ResNet18 标准预处理，取全局池化后的 512 维特征，计算各重建与原图的余弦相似度；另记录重建图与原图的 ResNet18 top-1 预测是否一致。top-1 一致表示模型预测保持相同，并不表示预测类别必然正确。

## 结果

| 重建方式 | 与原图特征余弦相似度均值 | 与原图 top-1 预测一致的图像比例 |
|---|---:|---:|
| 无丢包 VQGAN | {cos['clean_vqgan']:.4f} | {agree['clean_vqgan']:.0%} |
| 预训练 MaskGIT 一步恢复 | **{cos['pretrained_masked_class']:.4f}** | **{agree['pretrained_masked_class']:.0%}** |
| 空间填补 | {cos['spatial']:.4f} | {agree['spatial']:.0%} |

预训练方法与空间填补的逐图余弦相似度平均差为 **{analysis['pretrained_minus_spatial_mean_cosine']:+.4f}**，100 张中前者较高 **{analysis['pretrained_higher_images']}** 张、较低 **{analysis['pretrained_lower_images']}** 张。以图像为单位配对重采样 10,000 次，均值差的探索性 95% 区间为 **[{lo:+.4f}, {hi:+.4f}]**。在 top-1 一致性上，预训练方法一致而空间填补不一致的图像有 **{analysis['pretrained_agrees_and_spatial_disagrees']}** 张，反向有 **{analysis['spatial_agrees_and_pretrained_disagrees']}** 张。

## 边界与后续

这是一项**冻结 ResNet18 表示的代理指标**，不是论文 CLIP/LPIPS 评测，也不能直接证明图像语义完整保留。ResNet18 使用 ImageNet 预训练，而 Imagenette 是 ImageNet 子集，因此无法声称预训练数据严格不重叠。数据也已用于前面阶段开发，不是新盲测；上述重采样区间仅是开发集上的描述性不确定性。无丢包 VQGAN 行提供量化器自身造成变化的参照水平，不能直接用它替代原图。

后续仍需实现更贴近论文的语义或感知评测、完整控制协议和等发送资源比较；方法冻结后再准备新的项目独立留出集。交付包包括逐图 JSON、分析脚本、报告及 SHA-256 清单，不重新分发第三方大权重。
'''
out = out_dir / 'TokCom_第十七阶段图像特征相似度开发验证_2026-11-01.md'
out.write_text(content, encoding='utf-8')
print(out)
