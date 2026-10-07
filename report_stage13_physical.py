"""Report the completed Stage-13 explicit digital reference-link development sweep."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_stage13_physical import OUT, PROTOCOL, ROOT, key, plan, sha256

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    data = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    if not status["completed"] or data["test_accessed"] or data["used_for_training"]:
        raise ValueError("Physical development sweep is incomplete or violates isolation")
    if len(rows) != len(plan()) or {key(row) for row in rows} != {key(row) for row in plan()}:
        raise ValueError("Physical results do not exactly match the plan")
    figures = ROOT / "figures" / "stage13_physical"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"awgn": "#4c72b0", "rayleigh": "#c44e52"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for channel in ["awgn", "rayleigh"]:
        group = [row for row in data["summary"] if row["channel"] == channel]
        axes[0].errorbar([row["snr_esn0_db"] for row in group], [100 * row["measured_per_mean"] for row in group], yerr=[100 * row["measured_per_std"] for row in group], marker="o", capsize=4, label=channel.upper(), color=colors[channel])
        axes[1].errorbar([row["snr_esn0_db"] for row in group], [100 * row["recovered_all_token_accuracy_mean"] for row in group], yerr=[100 * row["recovered_all_token_accuracy_std"] for row in group], marker="o", capsize=4, label=channel.upper(), color=colors[channel])
    axes[0].set(title="CRC测得分组丢包率", xlabel="SNR Es/N0（dB）", ylabel="PER（%）", xticks=[0, 4, 8, 12, 16])
    axes[1].set(title="MaskGIT恢复后的全图token准确率", xlabel="SNR Es/N0（dB）", ylabel="全部token准确率（%）", xticks=[0, 4, 8, 12, 16])
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Stage-13显式CRC/FEC/16-QAM数字链路开发扫描", fontsize=15)
    fig.savefig(figures / "physical_sweep.png", dpi=180)
    plt.close(fig)

    metric_rows = []
    for row in data["summary"]:
        metric_rows.append([row["channel"].upper(), f"{row['snr_esn0_db']} dB", f"{100*row['measured_per_mean']:.2f}%", f"{100*row['payload_ber_mean']:.3f}%", f"{100*row['recovered_all_token_accuracy_mean']:.3f}%", "—" if row["maskgit_missing_accuracy_mean"] is None else f"{100*row['maskgit_missing_accuracy_mean']:.3f}%", "—" if row["spatial_missing_accuracy_mean"] is None else f"{100*row['spatial_missing_accuracy_mean']:.3f}%", row["undetected_wrong_tokens"]])
    report = f"""# Stage-13 TokCom显式数字物理层开发报告

本报告使用Stage-13按冻结开发规则选择的最佳ContextMaskGIT检查点，在同一150张开发验证图上评价。协议SHA-256为`{sha256(PROTOCOL)}`。由于该开发集也参与模型选择，结果只用于工程诊断，不是新的盲测或泛化估计；任何方法变化后的最终结论仍需另建留出集。

物理链路逐包执行10-bit token打包、CRC-16/CCITT-FALSE、K=7 `(171,133)` 八进制码率1/2卷积码、六个零终止位、单位平均能量Gray 16-QAM和软输入终止Viterbi。SNR定义为Es/N0；Rayleigh为逐复符号独立衰落并假设接收端完美CSI。这些是公开、可执行的参考配置，不是论文未披露物理层参数的精确复原。

{table(['信道', 'SNR', '实测PER', '载荷BER', '恢复后全token准确率', 'MaskGIT丢失位恢复率', '空间填补恢复率', '未检出错误token'], metric_rows)}

![显式数字链路扫描](figures/stage13_physical/physical_sweep.png)

每张图有256个源码token、每token 10 bit，共2,560源码bit。每4个token形成40-bit载荷；加入16-bit CRC、6个终止位和码率1/2卷积码后，每包124编码bit、31个16-QAM复符号，64包共7,936编码bit和1,984个复符号。交织器与包头开销未计入。

接收端只依据解码bit和CRC决定收到或擦除，不使用发送端token修补。CRC未检出的错误会作为错误token保留，因此报告同时给出只在擦除位置计算的恢复率，以及把未检出错误也计入的全token准确率。
"""
    (ROOT / "STAGE13_PHYSICAL_REPORT.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(rows), "groups": len(data["summary"])}))


if __name__ == "__main__":
    main()
