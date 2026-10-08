"""Create a scope-safe TokCom versus TaiChi resource and robustness comparison."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)


def validate_shared_grid(tok_rows, tai_rows, snrs=(0, 5, 10, 20)):
    """Fail closed unless both summaries exactly cover the preregistered grid."""
    expected = {(channel, float(snr)) for channel in ["awgn", "rayleigh"] for snr in snrs}
    tok = {(row["channel"], float(row["snr_esn0_db"])): row for row in tok_rows}
    tai = {(row["kind"], float(row["snr_db"])): row for row in tai_rows}
    if len(tok_rows) != len(expected) or len(tai_rows) != len(expected) or set(tok) != expected or set(tai) != expected:
        raise ValueError("The two paths do not share the frozen channel/SNR grid")
    if any(int(row["seeds"]) != 3 for row in list(tok_rows) + list(tai_rows)):
        raise ValueError("Every shared-grid point must aggregate exactly three seeds")
    return tok, tai


def main():
    budget = json.loads((ROOT / "COMMUNICATION_BUDGET.json").read_text(encoding="utf-8"))
    taichi = json.loads((ROOT / "STAGE12_FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    tokcom = json.loads((ROOT / "runs/stage13_physical_dev/summary.json").read_text(encoding="utf-8"))
    physical_status = json.loads((ROOT / "runs/stage13_physical_dev/status.json").read_text(encoding="utf-8"))
    if not physical_status["completed"] or tokcom["test_accessed"] or tokcom["used_for_training"]:
        raise ValueError("Stage-13 physical development sweep is incomplete or contaminated")
    snrs = [0, 5, 10, 20]
    tok_rows = tokcom["summary"]
    tai_rows = taichi["summary"]
    tok, tai = validate_shared_grid(tok_rows, tai_rows, snrs)
    ref = budget["configurations"]["tokcom_public_vqgan_reference_link"]
    sem = budget["configurations"]["taichi_stage10_stage11"]

    figures = ROOT / "figures" / "fair_comparison"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    bars = axes[0].bar(["TokCom\n图像重建", "TaiChi\n分类回答"], [ref["complex_channel_symbols"], sem["complex_channel_symbols"]], color=["#4c72b0", "#55a868"])
    axes[0].bar_label(bars, fmt="%.0f")
    axes[0].set(title="每样本复信道使用数", ylabel="复符号 / 复信道使用")
    for channel, color, marker in [("awgn", "#4c72b0", "o"), ("rayleigh", "#c44e52", "s")]:
        axes[1].errorbar(snrs, [100 * tai[(channel, snr)]["exact_match_mean"] for snr in snrs], yerr=[100 * tai[(channel, snr)]["exact_match_std"] for snr in snrs], label=channel.upper(), color=color, marker=marker, capsize=4)
        axes[2].errorbar(snrs, [100 * tok[(channel, snr)]["recovered_all_token_accuracy_mean"] for snr in snrs], yerr=[100 * tok[(channel, snr)]["recovered_all_token_accuracy_std"] for snr in snrs], label=channel.upper(), color=color, marker=marker, capsize=4)
    axes[1].set(title="TaiChi：完整回答准确率", xlabel="SNR（dB）", ylabel="回答准确率（%）", xticks=snrs)
    axes[2].set(title="TokCom：全图token准确率", xlabel="SNR Es/N0（dB）", ylabel="token准确率（%）", xticks=snrs)
    for axis in axes[1:]:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("共同信道网格下的资源与各自任务鲁棒性（指标不可互换）", fontsize=15)
    fig.savefig(figures / "resource_and_robustness.png", dpi=180)
    plt.close(fig)

    result_rows = []
    for channel in ["awgn", "rayleigh"]:
        for snr in snrs:
            a = tai[(channel, snr)]
            t = tok[(channel, snr)]
            result_rows.append([channel.upper(), f"{snr} dB", f"{100*a['exact_match_mean']:.2f}%", f"{100*t['recovered_all_token_accuracy_mean']:.3f}%", f"{100*t['measured_per_mean']:.2f}%"])
    report = f"""# TokCom 与 TaiChi/TokenCom 公平口径比较

## 可比较内容

两条路线统一使用AWGN/Rayleigh和0、5、10、20 dB Es/N0网格，每个条件三个随机种子；两者都把复符号归一化到单位平均能量，并假设接收端已知Rayleigh信道。通信资源统一换算为复信道使用数：TokCom参考数字链路每张256×256图使用{ref['complex_channel_symbols']}个16-QAM复符号；TaiChi连续语义信道每次图像/问题推理发送{sem['real_channel_values']}个实值，即{sem['complex_channel_symbols']}个复信道使用。TokCom/TaiChi比值为{ref['complex_channel_symbols']/sem['complex_channel_symbols']:.3f}。

![资源与鲁棒性](figures/fair_comparison/resource_and_robustness.png)

## 同一信道网格下的各自任务指标

{table(['信道', 'SNR', 'TaiChi完整回答准确率', 'TokCom全图token准确率', 'TokCom实测PER'], result_rows)}

## 不能直接比较的内容

TaiChi当前任务是Imagenette十分类问答，输出一个类别名称；TokCom任务是恢复256个离散图像token并可进一步解码图像。回答准确率和token准确率不是同一效用函数，不能用表中较大数值宣称某条路线更好。Stage-12 TaiChi数值来自300张预注册新留出图；Stage-13 TokCom物理扫描来自参与模型选择的150张开发图，因此数据证据等级也不同。

TokCom的1,984个复符号包含当前逐包CRC、卷积码和16-QAM，但未包含包头、同步、导频、信道估计、交织器信令和重传。TaiChi的512个复信道使用来自连续值I/Q配对，没有量化、熵编码、数字调制或FEC，且未计问题文本传输。因此3.875倍只能描述本工程当前参考配置的信道使用数，不能解释为端到端频谱效率比。

更严格的论文级公平实验仍需统一输入分辨率、任务输出、效用指标、信道估计开销、延迟、计算量和目标质量。可行方向是在同一图像集合上让两条路线都执行分类，或都执行图像/语义重建，并按相同复信道使用预算重新训练。
"""
    (ROOT / "TOKCOM_TAICHI_FAIR_COMPARISON.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    audit = {"shared_channels": ["awgn", "rayleigh"], "shared_snr_definition": "Es/N0 with unit mean complex-symbol energy", "shared_snr_db": snrs, "seeds_per_condition": 3, "tokcom_complex_symbols": ref["complex_channel_symbols"], "taichi_complex_channel_uses": sem["complex_channel_symbols"], "ratio_descriptive_only": ref["complex_channel_symbols"] / sem["complex_channel_symbols"], "metrics_not_interchangeable": True, "data_evidence_not_equal": True}
    (ROOT / "TOKCOM_TAICHI_FAIR_COMPARISON.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(result_rows), "ratio": audit["ratio_descriptive_only"]}))


if __name__ == "__main__":
    main()
