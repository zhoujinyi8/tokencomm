"""Create the preregistered Stage-12 final-test report and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_stage12_final import condition_key, planned_conditions
from prepare_stage12_holdout import ROOT, sha256


RUN = ROOT / "runs" / "stage12_final_test"
FIGURES = ROOT / "figures" / "stage12_final"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join(
        "| " + " | ".join(map(str, row)) + " |\n" for row in rows
    )


def main() -> None:
    protocol_path = ROOT / "STAGE12_FINAL_TEST_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    data = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = {condition_key(row) for row in planned_conditions(protocol)}
    actual = {condition_key(row) for row in rows}
    if not status["completed"] or not data["test_accessed"] or data["used_for_training"]:
        raise ValueError("Stage-12 completion/access invariant failed")
    if data["protocol_sha256"] != sha256(protocol_path) or actual != expected or len(rows) != len(expected):
        raise ValueError("Stage-12 results do not exactly match the frozen protocol")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    colors = {"awgn": "#4c72b0", "rayleigh": "#c44e52"}
    markers = {"awgn": "o", "rayleigh": "s"}
    for kind in protocol["channels"]:
        group = [row for row in data["summary"] if row["kind"] == kind]
        snr = [row["snr_db"] for row in group]
        axes[0].errorbar(
            snr,
            [100 * row["exact_match_mean"] for row in group],
            yerr=[100 * row["exact_match_std"] for row in group],
            label=kind.upper(), color=colors[kind], marker=markers[kind], capsize=4, linewidth=2,
        )
        axes[1].plot(
            snr, [row["loss_mean"] for row in group], label=kind.upper(),
            color=colors[kind], marker=markers[kind], linewidth=2,
        )
    axes[0].set(title="预注册最终测试准确率", xlabel="SNR (dB)", ylabel="完整回答准确率（%）", xticks=protocol["snr_db"])
    axes[0].set_ylim(max(0, min(100 * row["exact_match_mean"] for row in data["summary"]) - 3), 100.5)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set(title="预注册最终测试损失", xlabel="SNR (dB)", ylabel="回答损失", xticks=protocol["snr_db"])
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        f"Stage-12 BAN+KAN第{data['checkpoint_step']}步：{data['test_images']}张新独立测试图", fontsize=15
    )
    fig.savefig(FIGURES / "channel_sweep.png", dpi=180)
    plt.close(fig)

    controls = {row["variant"]: row for row in data["controls"]}
    control_order = ["correct", "wrong_class", "zero"]
    control_names = ["正确特征", "错配特征", "零特征"]
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bars = axis.bar(
        control_names,
        [100 * controls[key]["exact_match"] for key in control_order],
        color=["#4c72b0", "#dd8452", "#999999"],
    )
    axis.bar_label(bars, fmt="%.1f%%")
    axis.set(title="新最终测试视觉依赖控制", ylabel="完整回答准确率（%）", ylim=(0, 105))
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES / "controls.png", dpi=180)
    plt.close(fig)

    correct_rows = [row for row in rows if row["variant"] == "correct"]
    class_matrix = []
    for kind in protocol["channels"]:
        group = [row for row in correct_rows if row["kind"] == kind]
        class_matrix.append(
            [100 * np.mean([row["per_class_accuracy"][str(label)] for row in group]) for label in range(len(protocol["class_names"]))]
        )
    lower = max(0, min(min(row) for row in class_matrix) - 2)
    fig, axis = plt.subplots(figsize=(12, 3.4), layout="constrained")
    image = axis.imshow(class_matrix, vmin=lower, vmax=100, cmap="YlGn")
    axis.set_xticks(range(len(protocol["class_names"])), protocol["class_names"], rotation=25, ha="right")
    axis.set_yticks(range(len(protocol["channels"])), [kind.upper() for kind in protocol["channels"]])
    axis.set_title("跨全部SNR与种子的逐类别平均准确率（%）")
    for row in range(len(class_matrix)):
        for column in range(len(class_matrix[row])):
            axis.text(column, row, f"{class_matrix[row][column]:.1f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="准确率（%）")
    fig.savefig(FIGURES / "per_class.png", dpi=180)
    plt.close(fig)

    metric_rows = [
        [
            row["kind"].upper(), f'{row["snr_db"]:g} dB', f'{100 * row["exact_match_mean"]:.2f}%',
            f'{100 * row["exact_match_std"]:.2f}%', f'{row["loss_mean"]:.6f}', row["seeds"],
        ]
        for row in data["summary"]
    ]
    control_rows = [
        [control_names[index], f'{controls[key]["loss"]:.6f}', f'{100 * controls[key]["exact_match"]:.2f}%', controls[key]["count"]]
        for index, key in enumerate(control_order)
    ]
    error_union = sorted({index for row in correct_rows for index in row["error_indices"]})
    report = f"""# Stage-12 新留出集预注册最终测试报告

## 协议与隔离

本报告严格使用测试数据打开前提交的`STAGE12_FINAL_TEST_PROTOCOL.json`，协议SHA-256为`{data['protocol_sha256']}`。冻结结构为BAN+KAN，检查点第{data['checkpoint_step']}步、SHA-256为`{data['checkpoint_sha256']}`。新留出集共{data['test_images']}张、每类{protocol['per_class']}张；它排除了所有历史训练、验证和测试清单中的文件哈希与解码像素哈希。该数据已用于最终评价，但`used_for_training=false`，结果不得用于继续训练、调参或更换模型。

## 信道结果

{table(['信道', 'SNR', '准确率均值', '种子标准差', '平均损失', '种子数'], metric_rows)}

![Stage-12信道扫描](figures/stage12_final/channel_sweep.png)

## 视觉依赖控制

{table(['条件', '损失', '完整回答准确率', '图像数'], control_rows)}

![Stage-12视觉控制](figures/stage12_final/controls.png)

## 逐类别稳定性

![Stage-12逐类别准确率](figures/stage12_final/per_class.png)

全部预注册正确特征条件中共有{len(error_union)}张不同图片至少出现一次错误。逐条件错误索引保存在`runs/stage12_final_test/results.jsonl`，便于审计，但不得据此调整模型。

## 结论边界

这是当前独立实现的第二个预注册测试，不是作者模型的精确回放。视觉主干使用ImageNet预训练，而Imagenette是ImageNet子集，结果包含预训练分布重叠优势。每个信道条件使用{len(protocol['seeds'])}个随机种子；标准差描述信道随机性，不是数据抽样置信区间。新留出集现已消费，后续方法变化必须另建留出集。
"""
    (ROOT / "STAGE12_FINAL_TEST_REPORT.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(data["summary"]), "controls": len(data["controls"]), "test_images": data["test_images"]}))


if __name__ == "__main__":
    main()
