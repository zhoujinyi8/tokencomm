"""Write a compact four-hour progress snapshot for Stage 13."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "stage13_tokcom_large"
CACHE = ROOT / "data" / "imagenette_stage10" / "token_cache"


def main():
    protocol = json.loads((ROOT / "STAGE13_TOKCOM_PROTOCOL.json").read_text(encoding="utf-8"))
    rows = []
    history = RUN / "history.jsonl"
    if history.exists():
        rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
    latest = rows[-1] if rows else None
    validations = [row for row in rows if "validation" in row]
    cache_summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8")) if (CACHE / "summary.json").exists() else None
    cached_files = len(list(CACHE.glob("*.npy"))) if CACHE.exists() else 0
    complete = (RUN / "status.json").exists() and json.loads((RUN / "status.json").read_text(encoding="utf-8"))["completed"]
    stage14_protocol = (ROOT / "STAGE14_TOKCOM_FINAL_PROTOCOL.json").exists()
    stage14_materialized = (ROOT / "data/stage14_tokcom_final_holdout/identity.json").exists()
    stage14_status_path = ROOT / "runs/stage14_tokcom_final/status.json"
    stage14_complete = stage14_status_path.exists() and json.loads(stage14_status_path.read_text(encoding="utf-8"))["completed"]
    if complete and not (ROOT / "STAGE13_TOKCOM_REPORT.md").exists():
        from report_stage13 import main as write_final_report

        write_final_report()
    lines = [
        "# Stage-13 TokCom 大规模训练推进快照", "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 数据规模：{protocol['data']['training_images']} 训练图 / {protocol['data']['validation_images']} 验证图",
        f"- 训练预算：{protocol['training']['steps']} 步，batch={protocol['training']['batch_size']}，PER课程={protocol['training']['per_choices']}",
        f"- token缓存文件：{cached_files}/{protocol['data']['training_images'] + protocol['data']['validation_images']}",
        f"- token聚合缓存完成：{'是' if cache_summary else '否'}",
        f"- 已完成训练步：{latest['step'] if latest else 0}/{protocol['training']['steps']}",
        f"- 正式验证次数：{len(validations)}",
        f"- 最终完成：{'是' if complete else '否'}",
        "- Stage-13测试集访问：否",
        f"- Stage-14盲测协议冻结：{'是' if stage14_protocol else '否'}",
        f"- Stage-14新留出集物化：{'是' if stage14_materialized else '否'}",
        f"- Stage-14最终评价完成：{'是' if stage14_complete else '否'}", "",
    ]
    if latest:
        lines.extend(["## 最新训练状态", "", f"- loss：{latest['loss']:.6f}", f"- PER：{latest['per']}", f"- 学习率：{latest['lr']:.8f}", f"- 累计训练秒数：{latest['elapsed_seconds']:.1f}", ""])
    if validations:
        lines.extend(["## 冻结开发验证", "", "| 步数 | 平均交叉熵 | 丢失token恢复率 |", "|---:|---:|---:|"])
        for row in validations:
            metric = row["validation"]
            lines.append(f"| {row['step']} | {metric['mean_ce']:.6f} | {100*metric['mean_accuracy']:.3f}% |")
        lines.append("")
    if not cache_summary:
        lines.append("当前阶段正在构建公开VQGAN的256-token缓存；缓存器按图像原子写入，可在中断后继续。")
    elif not complete:
        next_validation = ((latest["step"] // protocol["training"]["validation_every"] + 1) * protocol["training"]["validation_every"]) if latest else protocol["training"]["validation_every"]
        lines.append(f"下一正式开发验证点：第{next_validation}步。")
    else:
        lines.append("训练已完成；详细结果位于 runs/stage13_tokcom_large/results.json。")
    (ROOT / "STAGE13_PROGRESS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"cached": cached_files, "step": latest["step"] if latest else 0, "complete": complete}))


if __name__ == "__main__":
    main()
