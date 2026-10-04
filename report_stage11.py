"""Build incremental and final evidence artifacts for the Stage-11 fusion ablation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "runs" / "stage11_report"
FIGURES = ROOT / "figures" / "stage11"
VARIANTS = (
    ("ban_kan", "BAN + KAN"),
    ("ban_mlp", "BAN + MLP"),
    ("concat_mlp", "Concat + MLP"),
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # A resumed run can repeat steps. The last persisted row is authoritative.
    by_step = {int(row["step"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


def collect():
    variants = []
    for key, label in VARIANTS:
        run = RUNS / f"stage11_ablation_{key}"
        cfg_path, status_path, results_path = run / "config.json", run / "status.json", run / "results.json"
        hist = read_history(run / "history.jsonl")
        checks = [row for row in hist if "validation" in row]
        cfg = read_json(cfg_path) if cfg_path.exists() else None
        status = read_json(status_path) if status_path.exists() else None
        results = read_json(results_path) if results_path.exists() else None
        variants.append(
            {
                "key": key,
                "label": label,
                "run": str(run.relative_to(ROOT)).replace("\\", "/"),
                "started": cfg is not None,
                "completed": bool(status and status.get("completed")),
                "step": hist[-1]["step"] if hist else 0,
                "target_steps": cfg["args"]["steps"] if cfg else 800,
                "latest_loss": hist[-1]["loss"] if hist else None,
                "elapsed_seconds": hist[-1].get("seconds") if hist else None,
                "config": cfg,
                "history": hist,
                "validations": checks,
                "status": status,
                "results": results,
            }
        )
    return variants


def best_validation(rows):
    if not rows:
        return None
    return min(rows, key=lambda row: (row["validation"]["loss"], -row["validation"]["exact_match"]))


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join(
        "| " + " | ".join(map(str, row)) + " |\n" for row in rows
    )


def write_progress(variants, now):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in variants:
        best = best_validation(item["validations"])
        if item["completed"]:
            state = "完成"
        elif item["started"]:
            state = "训练中"
        else:
            state = "等待自动接力"
        rows.append(
            [
                item["label"],
                state,
                f'{item["step"]}/{item["target_steps"]}',
                "—" if item["latest_loss"] is None else f'{item["latest_loss"]:.6f}',
                "—" if best is None else f'{best["validation"]["loss"]:.6f}',
                "—" if best is None else f'{100 * best["validation"]["exact_match"]:.2f}%',
            ]
        )
    active = next((x for x in variants if x["started"] and not x["completed"]), None)
    waiting = [x["label"] for x in variants if not x["started"]]
    if active:
        action = f'继续 {active["label"]} 至第 {active["target_steps"]} 步；每 200 步固定开发验证。'
    elif waiting:
        action = f'等待后台接力启动 {waiting[0]}。'
    else:
        action = "三组实验已结束，生成最终公平消融报告。"
    body = f"""# Stage-11 四小时推进快照

- 生成时间：{now}
- 协议：`STAGE11_ABLATION_PROTOCOL.json`
- 数据边界：仅使用训练集与固定开发验证集；Stage-10 最终测试集不参与结构选择。

{markdown_table(['结构', '状态', '步数', '最近训练损失', '当前最佳验证损失', '当前最佳准确率'], rows)}

下一动作：{action}

该快照是运行状态证据。只有 `status.json` 中 `completed=true` 才计为该结构完成；训练中间结果不用于提前改变冻结预算。
"""
    (ROOT / "STAGE11_PROGRESS.md").write_text(body, encoding="utf-8")
    serializable = []
    for item in variants:
        serializable.append({k: v for k, v in item.items() if k not in ("history", "config", "results")})
    snapshot = {"generated_at": now, "protocol": "STAGE11_ABLATION_PROTOCOL.json", "variants": serializable}
    (OUT / "latest_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "snapshots.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def plot_progress(variants):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    colors = ("#4c72b0", "#dd8452", "#55a868")
    any_history = False
    any_validation = False
    for (item, color) in zip(variants, colors):
        if item["history"]:
            any_history = True
            axes[0].plot(
                [row["step"] for row in item["history"]],
                [row["loss"] for row in item["history"]],
                label=item["label"], color=color, alpha=0.82,
            )
        if item["validations"]:
            any_validation = True
            axes[1].plot(
                [row["step"] for row in item["validations"]],
                [100 * row["validation"]["exact_match"] for row in item["validations"]],
                "o-", label=item["label"], color=color,
            )
    axes[0].set(title="等预算融合消融：训练损失", xlabel="优化步数", ylabel="回答损失")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)
    axes[1].set(title="固定均衡开发集", xlabel="优化步数", ylabel="完整回答准确率（%）", ylim=(0, 105))
    axes[1].grid(alpha=0.2)
    if any_history:
        axes[0].legend()
    if any_validation:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "首个验证点尚未到达", ha="center", va="center", transform=axes[1].transAxes)
    fig.suptitle("Stage-11：BAN、KAN 与简单融合的公平对比", fontsize=15)
    fig.savefig(FIGURES / "progress.png", dpi=170)
    plt.close(fig)


def write_final(variants):
    if not all(item["completed"] and item["results"] for item in variants):
        return False
    summary_rows, control_rows = [], []
    for item in variants:
        best = best_validation(item["validations"])
        cfg, results = item["config"], item["results"]
        controls = {row["variant"]: row for row in results["validation_controls"]}
        summary_rows.append(
            [
                item["label"], results["best_step"],
                f'{best["validation"]["loss"]:.6f}', f'{100 * best["validation"]["exact_match"]:.2f}%',
                f'{cfg["parameters"]["trainable_parameters"]:,}', f'{item["elapsed_seconds"] / 3600:.2f} h',
            ]
        )
        control_rows.append(
            [item["label"]]
            + [f'{100 * controls[key]["exact_match"]:.2f}%' for key in ("correct", "wrong_class", "zero")]
        )
    report = f"""# Stage-11：双视觉融合模块的等预算消融

## 实验问题

本阶段检验论文近似结构中的 BAN 和 KAN 是否在相同数据、随机种子、优化器、LoRA 配置、语义信道维度和 800 步训练预算下带来可测收益。比较 BAN+KAN、BAN+MLP、Concat+MLP 三种融合器。所有模型选择只使用固定的 100 张十类均衡开发图像。

![训练与开发验证曲线](figures/stage11/progress.png)

## 主要结果

{markdown_table(['结构', '最佳步数', '最佳验证损失', '完整回答准确率', '可训练参数', '训练耗时'], summary_rows)}

## 视觉依赖控制

{markdown_table(['结构', '正确视觉特征', '错配类别特征', '零视觉特征'], control_rows)}

正确特征高、错配特征与零特征低，才能说明回答依赖视觉输入，而不是只由问题模板或语言先验决定。具体预测和目标保留在各运行目录的 `results.json` 中。

## 结论边界

Stage-11 是开发集消融，不能当成新的盲测结果。Stage-10 最终测试集已被消费，本阶段没有再次读取它，也不使用它选择结构。视觉编码器使用 ImageNet 预训练权重，Imagenette 属于 ImageNet 子集，因此这些结果不能支持“视觉主干从未见过同分布数据”的说法。实现是论文结构的独立近似复现，不是作者权重或私有训练流程的精确回放。
"""
    (ROOT / "STAGE11_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    final = {
        "protocol": "STAGE11_ABLATION_PROTOCOL.json",
        "all_completed": True,
        "test_accessed": False,
        "variants": [
            {
                "name": item["key"],
                "best_step": item["results"]["best_step"],
                "best_validation": best_validation(item["validations"])["validation"],
                "trainable_parameters": item["config"]["parameters"]["trainable_parameters"],
                "elapsed_seconds": item["elapsed_seconds"],
                "controls": item["results"]["validation_controls"],
            }
            for item in variants
        ],
    }
    (OUT / "final_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", help="Timestamp written into the snapshot; defaults to local time")
    args = parser.parse_args()
    now = args.timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    variants = collect()
    write_progress(variants, now)
    plot_progress(variants)
    final = write_final(variants)
    print(json.dumps({"generated_at": now, "steps": {x["key"]: x["step"] for x in variants}, "final": final}))


if __name__ == "__main__":
    main()
