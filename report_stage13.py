"""Generate the audited Stage-13 TokCom development report and figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "stage13_tokcom_large"
FIGURES = ROOT / "figures" / "stage13_tokcom"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)


def load_completed():
    protocol_path = ROOT / "STAGE13_TOKCOM_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    result = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
    history = [json.loads(line) for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    steps = [int(row["step"]) for row in history]
    expected = list(range(1, protocol["training"]["steps"] + 1))
    validation_steps = [row["step"] for row in history if "validation" in row]
    expected_validation = list(range(protocol["training"]["validation_every"], protocol["training"]["steps"] + 1, protocol["training"]["validation_every"]))
    if not status["completed"] or not result["completed"]:
        raise ValueError("Stage 13 is not complete")
    if steps != expected or len(steps) != len(set(steps)):
        raise ValueError("Training history does not exactly cover the frozen budget")
    if validation_steps != expected_validation:
        raise ValueError("Validation points do not match the frozen protocol")
    if result["test_accessed"] or result["used_for_training"]:
        raise ValueError("Stage 13 must remain development-only")
    sweep_keys = {(float(row["per"]), int(row["seed"])) for row in result["sweep"]}
    expected_sweep = {(float(per), int(seed)) for per in protocol["training"]["per_choices"] for seed in protocol["validation"]["seeds"]}
    if sweep_keys != expected_sweep or len(result["sweep"]) != len(expected_sweep):
        raise ValueError("PER sweep does not match the frozen protocol")
    if {row["mode"] for row in result["controls"]} != {"correct", "wrong", "none", "spatial"}:
        raise ValueError("Context controls are incomplete")
    return protocol_path, protocol, status, result, history


def main():
    protocol_path, protocol, status, result, history = load_completed()
    FIGURES.mkdir(parents=True, exist_ok=True)
    steps = np.array([row["step"] for row in history])
    losses = np.array([row["loss"] for row in history])
    window = 100
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    validations = [row for row in history if "validation" in row]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axes[0].plot(steps, losses, color="#4c72b0", alpha=0.16, linewidth=0.7, label="逐步损失")
    axes[0].plot(steps[window - 1 :], smooth, color="#4c72b0", linewidth=2, label="100步移动平均")
    axes[0].set(title="8,000张图的混合PER训练", xlabel="优化步", ylabel="掩码token交叉熵")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot([row["step"] for row in validations], [row["validation"]["mean_ce"] for row in validations], marker="o", label="验证交叉熵")
    accuracy_axis = axes[1].twinx()
    accuracy_axis.plot([row["step"] for row in validations], [100 * row["validation"]["mean_accuracy"] for row in validations], color="#c44e52", marker="s", label="恢复率")
    axes[1].set(title="固定150张开发验证图、三种子", xlabel="优化步", ylabel="平均交叉熵")
    accuracy_axis.set_ylabel("丢失token精确恢复率（%）")
    axes[1].grid(alpha=0.2)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = accuracy_axis.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.suptitle("Stage-13 TokCom / ContextMaskGIT v2 训练与开发验证", fontsize=15)
    fig.savefig(FIGURES / "training.png", dpi=180)
    plt.close(fig)

    grouped = []
    for per in protocol["training"]["per_choices"]:
        rows = [row for row in result["sweep"] if float(row["per"]) == float(per)]
        grouped.append({"per": per, "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])), "accuracy_std": float(np.std([row["accuracy"] for row in rows])), "ce_mean": float(np.mean([row["ce"] for row in rows]))})
    fig, axis = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    axis.errorbar([100 * row["per"] for row in grouped], [100 * row["accuracy_mean"] for row in grouped], yerr=[100 * row["accuracy_std"] for row in grouped], marker="o", capsize=5, linewidth=2)
    axis.set(title="最佳检查点的PER鲁棒性（150张开发图、三种子）", xlabel="分组丢包率 PER（%）", ylabel="丢失token精确恢复率（%）", xticks=[10, 30, 50])
    axis.grid(alpha=0.25)
    fig.savefig(FIGURES / "per_sweep.png", dpi=180)
    plt.close(fig)

    controls = {row["mode"]: row for row in result["controls"]}
    order = ["correct", "wrong", "none", "spatial"]
    names = ["同图上下文", "跨类别错配", "全部遮蔽", "空间邻近填补"]
    fig, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    bars = axis.bar(names, [100 * controls[key]["accuracy"] for key in order], color=["#4c72b0", "#dd8452", "#999999", "#55a868"])
    axis.bar_label(bars, fmt="%.2f%%")
    axis.set(title="PER=30%的上下文依赖与非学习基线", ylabel="丢失token精确恢复率（%）")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES / "controls.png", dpi=180)
    plt.close(fig)

    previous = json.loads((ROOT / "runs/stage7_context_maskgit_v2_b16_fixed/results.json").read_text(encoding="utf-8"))["best"]["val"]
    old = {row["mode"]: row for row in previous}
    metric_rows = [[f"{100*row['per']:.0f}%", f"{100*row['accuracy_mean']:.3f}%", f"{100*row['accuracy_std']:.3f}%", f"{row['ce_mean']:.6f}", len(protocol["validation"]["seeds"])] for row in grouped]
    control_rows = [[names[index], f"{100*controls[key]['accuracy']:.3f}%", "—" if key == "spatial" else f"{controls[key]['ce']:.6f}", controls[key]["missing_tokens"]] for index, key in enumerate(order)]
    report = f"""# Stage-13：TokCom大数据ContextMaskGIT开发训练报告

## 范围与协议

本阶段沿第一篇TokCom论文的离散图像token路线推进。公开VQGAN把256×256图像转换为16×16、共256个离散token，码本大小1024；接收端采用本工程的ContextMaskGIT v2，通过二维位置嵌入、局部卷积块和全局Transformer恢复分组丢失token。它不输入类别、答案或其他CMI。

训练协议在运行前冻结，SHA-256为`{sha256(protocol_path)}`。训练使用{protocol['data']['training_images']}张图、{protocol['data']['validation_images']}张固定开发验证图、{protocol['training']['steps']}步、batch {protocol['training']['batch_size']}，在PER={protocol['training']['per_choices']}之间随机采样。最佳检查点按三种固定种子下的平均掩码token交叉熵选择。测试集未读取，`test_accessed=false`、`used_for_training=false`。

## 训练与模型选择

最佳检查点为第**{result['best_step']}**步，平均验证交叉熵为**{result['best_mean_ce']:.6f}**，丢失token精确恢复率为**{100*result['best_mean_accuracy']:.3f}%**。

![训练与验证曲线](figures/stage13_tokcom/training.png)

## PER扫描

{table(['PER', '恢复率均值', '种子标准差', '平均交叉熵', '种子数'], metric_rows)}

![PER扫描](figures/stage13_tokcom/per_sweep.png)

## 上下文控制与空间基线

{table(['输入条件', '恢复率', '掩码交叉熵', '丢失token数'], control_rows)}

![上下文控制](figures/stage13_tokcom/controls.png)

Stage-7使用1,000张训练图和2,000步时，开发验证集同图上下文恢复率为{100*old['correct']['accuracy']:.3f}%，空间邻近填补为{100*old['spatial']['accuracy']:.3f}%。本阶段扩大到8,000张和6,000步，二者的变化用于衡量扩充数据和混合PER课程是否改善泛化；由于训练种子、采样课程和预算同时变化，这不是单变量因果消融。

## 复现边界

本阶段使用与公开MaskGIT VQGAN匹配的离散码本和图像token规格，但接收端Transformer由本工程从随机参数训练，不是TokCom作者或Google原始预训练MaskGIT权重。Imagenette来自ImageNet，公开VQGAN也使用ImageNet预训练。指标只衡量丢失位置的token精确恢复和交叉熵，尚未加入论文中的CLIP、LPIPS、TCE、重传策略或跨模态上下文。因此这是可验证的近似实现，不是论文数值的精确回放。
"""
    (ROOT / "STAGE13_TOKCOM_REPORT.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    audit = {"audit_passed": True, "protocol_sha256": sha256(protocol_path), "history_steps": len(history), "unique_steps": len({row['step'] for row in history}), "validation_steps": [row['step'] for row in validations], "sweep_conditions": len(result["sweep"]), "controls": list(controls), "test_accessed": result["test_accessed"], "used_for_training": result["used_for_training"]}
    (ROOT / "STAGE13_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_step": result["best_step"], "sweep": len(result["sweep"]), "audit": True}))


if __name__ == "__main__":
    main()
