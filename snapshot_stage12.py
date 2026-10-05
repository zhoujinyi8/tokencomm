"""Write a lightweight progress snapshot for the frozen Stage-12 final test."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluate_stage12_final import condition_key, planned_conditions
from prepare_stage12_holdout import ROOT, sha256


def main() -> None:
    protocol_path = ROOT / "STAGE12_FINAL_TEST_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run = ROOT / "runs" / "stage12_final_test"
    results_path = run / "results.jsonl"
    rows = []
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    plan = planned_conditions(protocol)
    planned_keys = {condition_key(row) for row in plan}
    actual_keys = [condition_key(row) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or not set(actual_keys).issubset(planned_keys):
        raise ValueError("Stage-12 progress results violate frozen protocol")
    status = json.loads((run / "status.json").read_text(encoding="utf-8")) if (run / "status.json").exists() else None
    access = json.loads((run / "access.json").read_text(encoding="utf-8")) if (run / "access.json").exists() else None
    completed = len(rows)
    lines = [
        "# Stage-12 最终测试推进快照",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 协议SHA-256：`{sha256(protocol_path)}`",
        f"- 冻结模型：{protocol['selected_variant']}，第{protocol['stage11_selected_metrics']['best_step']}步",
        f"- 新留出集：{protocol['expected_test_images']}张，每类{protocol['per_class']}张",
        f"- 已完成条件：{completed}/{len(plan)}",
        f"- 测试已访问：{'是' if access and access.get('test_accessed') else '否'}",
        f"- 最终完成：{'是' if status and status.get('completed') else '否'}",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| 序号 | 视觉条件 | 信道 | SNR | 种子 | 准确率 | 损失 |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for index, row in enumerate(rows, 1):
            lines.append(
                f"| {index} | {row['variant']} | {row['kind'].upper()} | {row['snr_db']:g} dB | "
                f"{row['seed']} | {100 * row['exact_match']:.2f}% | {row['loss']:.6f} |"
            )
        lines.append("")
    if completed < len(plan):
        next_row = next(row for row in plan if condition_key(row) not in set(actual_keys))
        lines.append(
            f"下一条件：{next_row['variant']} / {next_row['kind'].upper()} / "
            f"{next_row['snr_db']:g} dB / seed {next_row['seed']}。"
        )
    else:
        lines.append("全部冻结条件已完成；等待或已经生成最终报告。")
    (ROOT / "STAGE12_PROGRESS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"completed": completed, "planned": len(plan), "final": bool(status and status.get("completed"))}))


if __name__ == "__main__":
    main()
