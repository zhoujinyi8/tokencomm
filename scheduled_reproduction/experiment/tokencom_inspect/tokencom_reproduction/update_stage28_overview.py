"""Advance the cumulative overview through the paired ten-image reconstruction."""
from pathlib import Path

outputs = Path(__file__).resolve().parents[3] / 'outputs'
source = outputs / 'TokCom_复现总览_截至2026-11-12_阶段26.md'
target = outputs / 'TokCom_复现总览_截至2026-11-12_阶段28.md'
text = source.read_text(encoding='utf-8')
text = text.replace('（截至 2026-11-12（阶段 26））', '（截至 2026-11-12，阶段 28）')
old = '| 第二十六阶段 | 作者 VQGAN 的十类开发图 token，AWGN 8 dB，每图五次 | 50 次共 3,200 包，数据错误 215 包、CRC 失败 452 包 | 开发集物理链路；尚未送入 MaskGIT 或计算图像指标 |'
new = old + '\n' + '\n'.join([
    '| 第二十七阶段 | 第 26 阶段的同一物理接收，先检查 2 张开发图 | 作者 GenDecoding 可以对真值比较及 CRC 遮蔽分别生成结果 | 单轮、单种子，仅单图功能检查 |',
    '| 第二十八阶段 | 第 26 阶段第 0 轮物理接收，10 张开发图，同一采样种子 | 真值比较遮蔽 2,389/2,560 token 正确，CRC 遮蔽 2,220/2,560；对干净 VQGAN 图的平均 MSE 0.00305/0.00750 | 两分支已知信息不同；仅开发图、单物理轮次；非论文指标 |',
])
assert old in text
text = text.replace(old, new)
text = text.replace('第十一至二十六阶段属于', '第十一至二十八阶段属于')
text = text.replace('1. 用作者 VQGAN 重新编码开发图并核对 MaskGIT 权重，再运行官方演示的 AWGN/CRC11、真实类别与多步生成；随后明确控制与 ACK 状态机，计入数据包头、ACK、导频和重试时延，并报告相同发送资源约束下的性能。',
                    '1. 对已保存的其余四轮作者物理接收重复十图配对重建，按图像聚合；之后明确控制与 ACK 状态机，计入数据包头、ACK、导频和重试时延，并报告相同发送资源约束下的性能。')
text = text.replace('第十至二十六阶段报告', '第十至二十八阶段报告')
text = text.replace('尚未送入 MaskGIT 或计算图像指标', '该阶段仅完成物理层，生成与 MSE 见阶段 27–28')
target.write_text(text, encoding='utf-8')
print(target)
stage29 = text.replace('（截至 2026-11-12，阶段 28）', '（截至 2026-11-12，阶段 29）')
anchor = '| 第二十八阶段 | 第 26 阶段第 0 轮物理接收，10 张开发图，同一采样种子 | 真值比较遮蔽 2,389/2,560 token 正确，CRC 遮蔽 2,220/2,560；对干净 VQGAN 图的平均 MSE 0.00305/0.00750 | 两分支已知信息不同；仅开发图、单物理轮次；非论文指标 |'
assert anchor in stage29
stage29 = stage29.replace(anchor, anchor + '\n| 第二十九阶段 | 固定一张开发图，复用第 26 阶段五轮 AWGN 8 dB 接收 | 真值比较分支正确 token 231–245/256；CRC 分支 220–228/256 | 同一张图的重复接收，仅波动抽查 |')
stage29 = stage29.replace('第十一至二十八阶段属于', '第十一至二十九阶段属于')
stage29 = stage29.replace('第十至二十八阶段报告', '第十至二十九阶段报告')
stage29 = stage29.replace('对已保存的其余四轮作者物理接收重复十图配对重建，按图像聚合', '对其余九张开发图补全五轮作者物理接收的配对重建，按图像聚合')
target29 = outputs / 'TokCom_复现总览_截至2026-11-12_阶段29.md'
target29.write_text(stage29, encoding='utf-8')
print(target29)
