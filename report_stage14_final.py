"""Generate the fail-closed Stage-14 TokCom blind-test report."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_stage13_physical import key, sha256


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE14_TOKCOM_FINAL_PROTOCOL.json"
OUT = ROOT / "runs/stage14_tokcom_final"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)


def load_completed():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    expected = {key(row) for row in protocol["conditions"]}
    actual = [key(row) for row in rows]
    if not status["completed"] or len(actual) != len(expected) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Stage-14 final test is incomplete or contains unplanned rows")
    if not summary["test_accessed"] or summary["used_for_training"] or any(not row["test_accessed"] or row["used_for_training"] for row in rows):
        raise ValueError("Final-test provenance flags are inconsistent")
    if summary["protocol_sha256"] != sha256(PROTOCOL_PATH) or summary["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
        raise ValueError("Final-test frozen identity mismatch")
    if len(summary["summary"]) != 8 or any(row["seeds"] != 3 for row in summary["summary"]):
        raise ValueError("Final summary grid is incomplete")
    return protocol, summary, rows


def main() -> None:
    protocol, data, rows = load_completed()
    figure_dir = ROOT / "figures/stage14_tokcom_final"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"awgn": "#4c72b0", "rayleigh": "#c44e52"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for channel in ["awgn", "rayleigh"]:
        group = [row for row in data["summary"] if row["channel"] == channel]
        snr = [row["snr_esn0_db"] for row in group]
        axes[0].errorbar(snr, [100 * row["measured_per_mean"] for row in group], yerr=[100 * row["measured_per_std"] for row in group], marker="o", capsize=4, label=channel.upper(), color=colors[channel])
        axes[1].errorbar(snr, [100 * row["recovered_all_token_accuracy_mean"] for row in group], yerr=[100 * row["recovered_all_token_accuracy_std"] for row in group], marker="o", capsize=4, label=channel.upper(), color=colors[channel])
        axes[2].plot(snr, [np.nan if row["maskgit_missing_accuracy_mean"] is None else 100 * row["maskgit_missing_accuracy_mean"] for row in group], marker="o", label=f"{channel.upper()} MaskGIT", color=colors[channel])
        axes[2].plot(snr, [np.nan if row["spatial_missing_accuracy_mean"] is None else 100 * row["spatial_missing_accuracy_mean"] for row in group], marker="x", linestyle="--", label=f"{channel.upper()} 空间填补", color=colors[channel])
    axes[0].set(title="实测分组丢包率", ylabel="PER（%）")
    axes[1].set(title="恢复后全图token准确率", ylabel="准确率（%）")
    axes[2].set(title="丢失位置恢复：模型与基线", ylabel="丢失token准确率（%）")
    for axis in axes:
        axis.set(xlabel="Es/N0（dB）", xticks=[0, 5, 10, 20])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Stage-14 TokCom：300张新留出图的预注册盲测", fontsize=15)
    fig.savefig(figure_dir / "channel_sweep.png", dpi=180)
    plt.close(fig)

    percent = lambda value: "—" if value is None else f"{100*value:.3f}%"
    metric_rows = [[row["channel"].upper(), f"{row['snr_esn0_db']} dB", f"{100*row['measured_per_mean']:.2f}%", f"{100*row['payload_ber_mean']:.3f}%", f"{100*row['recovered_all_token_accuracy_mean']:.3f}%", percent(row["maskgit_missing_accuracy_mean"]), percent(row["spatial_missing_accuracy_mean"]), row["undetected_wrong_tokens"]] for row in data["summary"]]
    report = f"""# Stage-14 TokCom预注册盲测报告

本报告在打开新留出集前冻结最佳Stage-13检查点、实现文件哈希、数据选择规则、24个信道条件和全部指标。测试集含{protocol['expected_test_images']}张公开Imagenette图像，每类{protocol['per_class']}张；它与所有既有训练、开发、历史测试及Stage-12留出图在文件哈希和解码像素哈希上零重叠。协议SHA-256为`{sha256(PROTOCOL_PATH)}`，检查点SHA-256为`{protocol['checkpoint_sha256']}`。

{table(['信道', 'Es/N0', '实测PER', '载荷BER', '全图token准确率', 'MaskGIT丢失位恢复率', '空间填补恢复率', '未检出错误token'], metric_rows)}

![Stage-14盲测信道扫描](figures/stage14_tokcom_final/channel_sweep.png)

每个点使用同一300张图和三个独立信道种子；标准差反映信道随机性，不代表900张独立图。链路为每token 10 bit、每包4 token、CRC-16、K=7 `(171,133)` 码率1/2卷积码、六零终止、Gray 16-QAM和软Viterbi；Rayleigh假设逐符号独立衰落及完美接收端CSI。

这是本工程ContextMaskGIT接收端的盲测。它仍是对TokCom论文的近似实现：使用公开ImageNet VQGAN，但恢复Transformer由本工程训练；没有作者MaskGIT权重、跨模态CMI、重传/TCE、包头/同步/导频开销，也没有CLIP、LPIPS或人类语义质量指标。`test_accessed=true`仅表示按冻结协议完成最终评价，`used_for_training=false`。
"""
    (ROOT / "STAGE14_TOKCOM_FINAL_REPORT.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    audit = {
        "audit_passed": True,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "test_manifest_sha256": data["test_manifest_sha256"],
        "test_images": data["test_images"],
        "conditions": len(rows),
        "unique_conditions": len({key(row) for row in rows}),
        "groups": len(data["summary"]),
        "test_accessed": True,
        "used_for_training": False,
        "known_gaps": ["author MaskGIT weights", "cross-modal CMI", "retransmission/TCE", "full physical overhead", "CLIP/LPIPS"],
    }
    (ROOT / "STAGE14_TOKCOM_FINAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(rows), "audit": True}))


if __name__ == "__main__":
    main()
