"""Build incremental and final evidence artifacts for the Stage-11 fusion ablation."""
from __future__ import annotations

import argparse
import collections
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


def read_raw_history(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect():
    variants = []
    for key, label in VARIANTS:
        run = RUNS / f"stage11_ablation_{key}"
        cfg_path, status_path, results_path = run / "config.json", run / "status.json", run / "results.json"
        latest_validation_path = run / "latest_validation.json"
        raw_hist = read_raw_history(run / "history.jsonl")
        hist = read_history(run / "history.jsonl")
        checks = [row for row in hist if "validation" in row]
        cfg = read_json(cfg_path) if cfg_path.exists() else None
        status = read_json(status_path) if status_path.exists() else None
        results = read_json(results_path) if results_path.exists() else None
        latest_validation = read_json(latest_validation_path) if latest_validation_path.exists() else None
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
                "raw_history": raw_hist,
                "validations": checks,
                "status": status,
                "results": results,
                "latest_validation": latest_validation,
                "latest_validation_mtime": latest_validation_path.stat().st_mtime if latest_validation_path.exists() else None,
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


def validation_diagnostics(variants):
    candidates = [item for item in variants if item["latest_validation"] and item["validations"]]
    if not candidates:
        return None
    item = max(candidates, key=lambda row: row["latest_validation_mtime"])
    metrics = item["latest_validation"]
    manifest = read_json(ROOT / "data" / "imagenette_stage10" / "manifest.json")
    classes = [name.lower() for name in manifest["class_names"]]
    index = {name: i for i, name in enumerate(classes)}
    matrix = [[0 for _ in range(len(classes) + 1)] for _ in classes]
    for prediction, target in zip(metrics["predictions"], metrics["targets"]):
        target_index = index[target.lower()]
        prediction_index = index.get(prediction.lower(), len(classes))
        matrix[target_index][prediction_index] += 1
    per_class = []
    for i, name in enumerate(classes):
        count = sum(matrix[i])
        correct = matrix[i][i]
        per_class.append({"class": name, "count": count, "correct": correct, "accuracy": correct / count if count else 0.0})
    return {
        "variant": item["key"],
        "label": item["label"],
        "step": item["validations"][-1]["step"],
        "loss": metrics["loss"],
        "exact_match": metrics["exact_match"],
        "classes": classes,
        "matrix": matrix,
        "per_class": per_class,
    }


def plot_diagnostics(diagnostics):
    if diagnostics is None:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    classes = diagnostics["classes"]
    short = [name.replace("english ", "").replace("cassette ", "cassette\n") for name in classes]
    accuracies = [100 * row["accuracy"] for row in diagnostics["per_class"]]
    matrix = diagnostics["matrix"]
    include_other = any(row[-1] for row in matrix)
    shown = [row if include_other else row[:-1] for row in matrix]
    predicted = short + (["其他"] if include_other else [])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    bars = axes[0].bar(short, accuracies, color="#4c72b0")
    axes[0].bar_label(bars, fmt="%.0f%%")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(ylabel="完整回答准确率（%）", ylim=(0, 108), title="逐类别准确率")
    axes[0].grid(axis="y", alpha=0.2)
    image = axes[1].imshow(shown, cmap="Blues", vmin=0)
    axes[1].set_xticks(range(len(predicted)), predicted, rotation=45, ha="right")
    axes[1].set_yticks(range(len(short)), short)
    axes[1].set(xlabel="预测类别", ylabel="真实类别", title="回答混淆矩阵（计数）")
    for row in range(len(shown)):
        for col in range(len(shown[row])):
            value = shown[row][col]
            if value:
                axes[1].text(col, row, str(value), ha="center", va="center", color="white" if value > 5 else "black")
    fig.colorbar(image, ax=axes[1], shrink=0.8)
    fig.suptitle(
        f"Stage-11 {diagnostics['label']} 第{diagnostics['step']}步："
        f"{100 * diagnostics['exact_match']:.1f}% / loss {diagnostics['loss']:.4f}", fontsize=14,
    )
    fig.savefig(FIGURES / "latest_diagnostics.png", dpi=170)
    plt.close(fig)


def write_progress(variants, now, diagnostics):
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
    diagnostic_section = ""
    if diagnostics:
        class_rows = [
            [row["class"], row["correct"], row["count"], f'{100 * row["accuracy"]:.1f}%']
            for row in diagnostics["per_class"]
        ]
        diagnostic_section = f"""
## 最近一次固定开发验证

结构：{diagnostics['label']}；步数：{diagnostics['step']}；损失：{diagnostics['loss']:.6f}；完整回答准确率：{100 * diagnostics['exact_match']:.2f}%。

{markdown_table(['类别', '正确', '样本', '准确率'], class_rows)}

![逐类别准确率与混淆矩阵](figures/stage11/latest_diagnostics.png)
"""
    body = f"""# Stage-11 四小时推进快照

- 生成时间：{now}
- 协议：`STAGE11_ABLATION_PROTOCOL.json`
- 数据边界：仅使用训练集与固定开发验证集；Stage-10 最终测试集不参与结构选择。

{markdown_table(['结构', '状态', '步数', '最近训练损失', '当前最佳验证损失', '当前最佳准确率'], rows)}

下一动作：{action}

该快照是运行状态证据。只有 `status.json` 中 `completed=true` 才计为该结构完成；训练中间结果不用于提前改变冻结预算。
{diagnostic_section}
"""
    (ROOT / "STAGE11_PROGRESS.md").write_text(body.rstrip() + "\n", encoding="utf-8")
    serializable = []
    for item in variants:
        serializable.append({k: v for k, v in item.items() if k not in ("history", "raw_history", "config", "results")})
    snapshot = {"generated_at": now, "protocol": "STAGE11_ABLATION_PROTOCOL.json", "variants": serializable,
                "latest_diagnostics": diagnostics}
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


def recovery_audit(variants):
    audited = []
    for item in variants:
        groups = collections.defaultdict(list)
        for row in item["raw_history"]:
            groups[int(row["step"])].append(row)
        expected = set(range(1, item["target_steps"] + 1))
        missing = sorted(expected - set(groups))
        extra = sorted(set(groups) - expected)
        duplicates = {step: rows for step, rows in groups.items() if len(rows) > 1}
        spreads = {
            step: max(row["loss"] for row in rows) - min(row["loss"] for row in rows)
            for step, rows in duplicates.items()
        }
        duplicate_validation_steps = [
            step for step, rows in duplicates.items() if sum("validation" in row for row in rows) > 1
        ]
        validation_steps = [row["step"] for row in item["validations"]]
        expected_validations = list(range(200, item["target_steps"] + 1, 200))
        passed = (
            not missing and not extra and len(groups) == item["target_steps"]
            and max(spreads.values(), default=0.0) == 0.0
            and not duplicate_validation_steps and validation_steps == expected_validations
            and item["completed"] and item["results"] and item["results"].get("test_accessed") is False
        )
        audited.append(
            {
                "variant": item["key"],
                "passed": passed,
                "raw_rows": len(item["raw_history"]),
                "unique_steps": len(groups),
                "missing_steps": missing,
                "extra_steps": extra,
                "duplicate_step_count": len(duplicates),
                "duplicate_row_count": sum(len(rows) - 1 for rows in duplicates.values()),
                "duplicate_step_range": [min(duplicates), max(duplicates)] if duplicates else None,
                "max_duplicate_loss_spread": max(spreads.values(), default=0.0),
                "duplicate_validation_steps": duplicate_validation_steps,
                "validation_steps": validation_steps,
                "test_accessed": item["results"].get("test_accessed") if item["results"] else None,
            }
        )
    return {"all_passed": all(row["passed"] for row in audited), "variants": audited}


def plot_final_comparison(variants):
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in variants]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    best = [best_validation(item["validations"])["validation"] for item in variants]
    controls = [
        {row["variant"]: row for row in item["results"]["validation_controls"]}
        for item in variants
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    bars = axes[0].bar(labels, [row["loss"] for row in best], color=colors)
    axes[0].bar_label(bars, fmt="%.4f")
    axes[0].set(title="最佳开发验证损失", ylabel="回答损失", yscale="log")
    axes[0].grid(axis="y", alpha=0.2)
    bars = axes[1].bar(labels, [100 * row["exact_match"] for row in best], color=colors)
    axes[1].bar_label(bars, fmt="%.1f%%")
    axes[1].set(title="最佳完整回答准确率", ylabel="准确率（%）", ylim=(90, 101.5))
    axes[1].grid(axis="y", alpha=0.2)
    x = list(range(len(labels)))
    width = 0.24
    for offset, key, name, color in [(-width, "correct", "正确特征", "#4c72b0"),
                                      (0, "wrong_class", "错配特征", "#dd8452"),
                                      (width, "zero", "零特征", "#999999")]:
        values = [100 * control[key]["exact_match"] for control in controls]
        axes[2].bar([value + offset for value in x], values, width, label=name, color=color)
    axes[2].set_xticks(x, labels)
    axes[2].set(title="最佳检查点视觉依赖控制", ylabel="准确率（%）", ylim=(0, 105))
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.2)
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Stage-11 等预算融合消融最终比较", fontsize=15)
    fig.savefig(FIGURES / "final_comparison.png", dpi=170)
    plt.close(fig)


def write_final(variants):
    if not all(item["completed"] and item["results"] for item in variants):
        return False
    audit = recovery_audit(variants)
    if not audit["all_passed"]:
        raise RuntimeError("Stage-11 recovery audit failed; refusing final report")
    (ROOT / "STAGE11_RECOVERY_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_final_comparison(variants)
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
    winner = min(variants, key=lambda item: (
        best_validation(item["validations"])["validation"]["loss"],
        -best_validation(item["validations"])["validation"]["exact_match"],
    ))
    winner_best = best_validation(winner["validations"])["validation"]
    audit_rows = [
        [
            next(item["label"] for item in variants if item["key"] == row["variant"]),
            row["raw_rows"], row["unique_steps"], row["duplicate_row_count"],
            f'{row["max_duplicate_loss_spread"]:.1f}', "通过" if row["passed"] else "失败",
        ]
        for row in audit["variants"]
    ]
    report = f"""# Stage-11：双视觉融合模块的等预算消融

## 实验问题

本阶段检验论文近似结构中的 BAN 和 KAN 是否在相同数据、随机种子、优化器、LoRA 配置、语义信道维度和 800 步训练预算下带来可测收益。比较 BAN+KAN、BAN+MLP、Concat+MLP 三种融合器。所有模型选择只使用固定的 100 张十类均衡开发图像。

![训练与开发验证曲线](figures/stage11/progress.png)

## 主要结果

{markdown_table(['结构', '最佳步数', '最佳验证损失', '完整回答准确率', '可训练参数', '训练耗时'], summary_rows)}

![最终融合消融比较](figures/stage11/final_comparison.png)

按预先冻结的“最低验证损失，其次最高完整回答准确率”规则，胜出结构为**{winner['label']}**：第{winner['results']['best_step']}步验证损失{winner_best['loss']:.6f}、准确率{100 * winner_best['exact_match']:.2f}%。相较BAN+MLP，它增加{winner['config']['parameters']['trainable_parameters'] - variants[1]['config']['parameters']['trainable_parameters']:,}个可训练参数；相较Concat+MLP增加{winner['config']['parameters']['trainable_parameters'] - variants[2]['config']['parameters']['trainable_parameters']:,}个。

## 视觉依赖控制

{markdown_table(['结构', '正确视觉特征', '错配类别特征', '零视觉特征'], control_rows)}

正确特征高、错配特征与零特征低，才能说明回答依赖视觉输入，而不是只由问题模板或语言先验决定。具体预测和目标保留在各运行目录的 `results.json` 中。

## 恢复轨迹审计

{markdown_table(['结构', '原始日志行', '唯一训练步', '重算行', '重算损失最大差异', '结论'], audit_rows)}

BAN+KAN和BAN+MLP在任务计划恢复时重算了最近检查点之后的少量步骤；所有重算步损失逐值一致，验证点没有重复，去重后均完整覆盖1–800。Concat+MLP没有重复步骤。详细机器可读证据见`STAGE11_RECOVERY_AUDIT.json`。

## 结论边界

Stage-11 是开发集消融，不能当成新的盲测结果。三种结构只运行一个训练随机种子，且开发集只有100张图；1个百分点的准确率差异不足以单独支持统计显著性结论。较低验证损失支持在当前配置下选择BAN+KAN，但不能证明BAN或KAN在其他任务、数据和信道中必然更优。Stage-10最终测试集已被消费，本阶段没有再次读取它，也不使用它选择结构。视觉编码器使用ImageNet预训练权重，Imagenette属于ImageNet子集，因此这些结果不能支持“视觉主干从未见过同分布数据”的说法。实现是论文结构的独立近似复现，不是作者权重或私有训练流程的精确回放。
"""
    (ROOT / "STAGE11_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    final = {
        "protocol": "STAGE11_ABLATION_PROTOCOL.json",
        "all_completed": True,
        "selected_variant": winner["key"],
        "selection_rule": "minimum validation loss, then maximum exact match",
        "recovery_audit": audit,
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
    diagnostics = validation_diagnostics(variants)
    plot_diagnostics(diagnostics)
    write_progress(variants, now, diagnostics)
    plot_progress(variants)
    final = write_final(variants)
    print(json.dumps({"generated_at": now, "steps": {x["key"]: x["step"] for x in variants}, "final": final}))


if __name__ == "__main__":
    main()
