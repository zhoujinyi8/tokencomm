"""Generate the tracked communication-budget JSON and Markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from communication_budget import default_report


ROOT = Path(__file__).resolve().parent


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def main() -> None:
    report = default_report()
    configs = report["configurations"]
    ref = configs["tokcom_public_vqgan_reference_link"]
    tiny = configs["tokcom_tiny_baseline"]
    taichi = configs["taichi_stage10_stage11"]
    figure_dir = ROOT / "figures" / "communication_budget"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    labels = ["TokCom\n公开VQGAN规格", "TokCom\n小型基线", "TaiChi\nStage-10/11"]
    symbols = [ref["complex_channel_symbols"], tiny["complex_channel_symbols"], taichi["complex_channel_symbols"]]
    per_pixel = [ref["complex_symbols_per_pixel"], tiny["complex_symbols_per_pixel"], taichi["complex_symbols_per_pixel"]]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    bars = axes[0].bar(labels, symbols, color=colors)
    axes[0].bar_label(bars, fmt="%.0f")
    axes[0].set(title="每个样本的复信道符号", ylabel="复符号数")
    axes[0].grid(axis="y", alpha=0.2)
    bars = axes[1].bar(labels, per_pixel, color=colors)
    axes[1].bar_label(bars, fmt="%.4f")
    axes[1].set(title="按视觉输入像素归一化", ylabel="复符号/源像素")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("通信开销的可执行核算（任务与输出不同，仅作描述）", fontsize=14)
    fig.savefig(figure_dir / "channel_uses.png", dpi=170)
    plt.close(fig)
    (ROOT / "COMMUNICATION_BUDGET.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# TokCom 与 TaiChi 的统一通信开销核算

## 为什么分开报告三种单位

离散VQ token的源码率、加入CRC/FEC后的编码比特数、以及连续语义向量占用的复信道符号是不同量。连续语义向量没有指定量化器和熵编码时，不存在可验证的“比特数”，因此本报告不把TaiChi的512个复符号换算成任意bpp。

| 配置 | 任务 | 源表示 | 源码率 | 分包 | 编码比特 | 复信道符号 | 每源像素复符号 |
|---|---|---:|---:|---:|---:|---:|---:|
| 公开VQGAN规格 + 当前TokCom数字链路 | 图像重建 | {ref['num_tokens']}×{ref['token_bits']}={ref['source_bits']} bit | {fmt(ref['source_bits_per_pixel'])} bpp | {ref['packets']}×最多{ref['packet_tokens']} token | {ref['coded_bits']} | {ref['complex_channel_symbols']} | {fmt(ref['complex_symbols_per_pixel'])} |
| 小型TokCom基线 | 32×32图像重建 | {tiny['num_tokens']}×{tiny['token_bits']}={tiny['source_bits']} bit | {fmt(tiny['source_bits_per_pixel'])} bpp | {tiny['packets']}×最多{tiny['packet_tokens']} token | {tiny['coded_bits']} | {tiny['complex_channel_symbols']} | {fmt(tiny['complex_symbols_per_pixel'])} |
| Stage-10/11 TaiChi | Imagenette问答 | {taichi['visual_tokens']}×{taichi['real_channel_dimension_per_token']}={taichi['real_channel_values']}个实数 | 未定义 | 连续语义信道 | 未定义 | {taichi['complex_channel_symbols']} | {fmt(taichi['complex_symbols_per_pixel'])} |

![通信开销核算](figures/communication_budget/channel_uses.png)

## TokCom数字链路的精确边界

公开VQGAN规格使用256个、1024码本的离散token，源码率为256×10/256²={fmt(ref['source_bits_per_pixel'])} bpp。当前参考链路每4个token形成一个40 bit载荷，逐包加入16 bit CRC与6个卷积码终止位，再用码率1/2卷积码和16-QAM发送。因此每包产生124个编码比特和31个复符号，64包共{ref['coded_bits']}个编码比特和{ref['complex_channel_symbols']}个复符号。包含CRC与终止位后的编码比特/源码比特为{fmt(ref['coded_bits_per_source_bit'], 4)}。

此数字不包含包头、交织器描述、同步、导频、信道估计或重传。它是当前`digital_link.py`的可执行参考配置，不代表论文未披露的全部物理层细节。测试会把解析计算结果与无噪声链路实际统计逐项比较。

## TaiChi语义信道的精确边界

Stage-10/11模型发送16个视觉token，每个token由896维LLM空间压缩成64个实数。`Channel`把相邻两个实数配成一个I/Q复符号，因此每张图占用16×64/2={taichi['complex_channel_symbols']}个复信道符号。归一化到224×224视觉输入为{fmt(taichi['complex_symbols_per_pixel'])}个复符号/像素。

该计数不包含问题文本的传输、同步、导频、信道估计、模型分发和计算成本。连续值没有量化与数字链路，因此不能报告编码比特或bpp。

## 公平解释

当前参考配置下，TokCom数字链路的复符号数是TaiChi视觉语义信道的{fmt(report['descriptive_symbol_ratio']['tokcom_reference_over_taichi'], 3)}倍。这个比例只能描述本工程的信道使用量，不能证明TaiChi更高效：TokCom输出重建图像，而Stage-10/11 TaiChi只回答十分类问题。论文级公平对比还需要统一任务、质量指标、信道模型、文本/侧信息开销和目标输出。
"""
    (ROOT / "COMMUNICATION_BUDGET_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"json": "COMMUNICATION_BUDGET.json", "report": "COMMUNICATION_BUDGET_REPORT.md"}))


if __name__ == "__main__":
    main()
