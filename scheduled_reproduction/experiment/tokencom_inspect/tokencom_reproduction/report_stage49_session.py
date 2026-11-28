"""Record a frozen-evaluation continuation without partial performance means."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
outputs = root.parents[2] / 'outputs'
progress = json.loads((run / 'stage46_recovery_progress.json').read_text(encoding='utf-8'))
metrics = json.loads((run / 'stage47_metric_file_audit.json').read_text(encoding='utf-8'))
prior_sync = json.loads((run / 'stage46_github_sync_receipt.json').read_text(encoding='utf-8'))
current = progress['completed_recovery_cases']
metric_count = metrics['metric_receptions_saved']
recovery_audit = json.loads((run / 'stage46_output_audit.json').read_text(encoding='utf-8'))
assert recovery_audit['all_checks_passed'] and recovery_audit['checked_cases'] == current
assert metrics['all_checks_passed'] and 11 <= metric_count <= current <= 500
assert not (run / 'stage47_complete_holdout_summary.json').exists()
result = {
    'session_start_utc': '2026-11-22T04:26:32.211Z',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'initial_recovery_cases': 245,
    'initial_metric_cases': 11,
    'recovery_cases': current,
    'new_recovery_cases': current - 245,
    'complete_five_trial_images': progress['complete_five_trial_images'],
    'metric_cases': metric_count,
    'new_metric_cases': metric_count - 11,
    'expected_cases': 500,
    'paths_per_metric_case': 6,
    'model_performance_aggregated': False,
    'frozen_scientific_settings_changed': False,
    'incomplete_aggregation_guard_checked': True,
    'incomplete_aggregation_guard_result': 'ValueError: Incomplete frozen evaluation: aggregate withheld',
    'prior_verified_ci': {
        key: prior_sync[key] for key in (
            'commit', 'ci_verified_run', 'latest_commit_ci',
            'unit_tests_passed', 'unit_tests_skipped', 'asset_integrations_deselected')
    },
    'current_publication_receipt': 'runs/stage46_github_sync_receipt.json',
}
(run / 'stage49_session_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
report = outputs / 'TokCom_第四十九阶段冻结留出持续执行_2026-11-22.md'
report.write_text(f'''# TokCom 第四十九阶段：冻结留出持续执行

本轮从 245 组恢复、11 组指标缓存继续执行。当前恢复累计 {current}/500 组，本轮增加 {current-245} 组；五轮信道均完成的图像 {progress['complete_five_trial_images']}/100 张。作者 token 编码 100 张、物理接收缓存 500 组已在此前完成。

逐组图像指标累计 {metric_count}/500 组，本轮增加 {metric_count-11} 组；每组保存直接解码、真值掩码恢复、CRC 恢复和三档软门控六条路径，共 {metric_count*6} 条路径记录。所有指标组均通过身份与有限值审计。部分缓存未用于估计留出集整体性能；最终汇总入口要求 100 张图像、每张五轮信道和全部六条路径齐全。

## 执行与可复核性

恢复和指标继续沿用原有源码、协议、模型权重及执行锁，没有因留出结果修改种子、掩码、门控比例或指标定义。恢复审计核对输入身份、未掩码位置、码本范围和软门控构造。指标审计核对来源文件、锁、六条路径与有限值。报告只公开执行数量，生成缓存与权重保留本地。

本轮重新核验全部 100 个 token 文件和 500 个物理接收文件，审计通过；作者仓库固定提交和 10 个实现文件身份保持一致。另实际运行全量汇总入口，因指标缺组按预期抛出 `Incomplete frozen evaluation: aggregate withheld`，没有生成正式均值。

本轮开始时 GitHub 已核验提交 `{prior_sync['commit']}`，检查运行 {prior_sync['ci_verified_run']} 成功：96 项单元测试通过、2 项跳过、11 项依赖实验资产的集成测试未运行。此检查不代表完整论文结果或历史冻结资产全部通过。本轮发布后的实际提交与检查状态另记入同步回执。

继续更新同一草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。本轮未新增模型实现或调整科学配置，重点是按冻结协议补齐恢复与图像指标证据。

## 剩余工作

恢复剩余 {500-current} 组，图像指标剩余 {500-metric_count} 组。全部完成后按图像先平均五轮信道，再汇总 100 张图像。当前仅为 8 dB、固定生成种子及真实类别假设下的项目留出评估；论文 TCE、反馈协议及多 SNR 复现仍未完成。该留出集与项目先前图像文件及像素去重，不宣称与公共预训练数据完全独立。
''', encoding='utf-8')
names = [
    'report_stage49_session.py', 'audit_stage46_outputs.py',
    'audit_stage47_metric_files.py', 'runs/stage49_session_progress.json',
    'runs/stage46_recovery_progress.json', 'runs/stage46_output_audit.json',
    'runs/stage46_recovery_lock.json', 'runs/stage47_metrics_lock.json',
    'runs/stage47_metric_file_audit.json',
]
items = [(name, root) for name in names] + [(report.name, outputs)]
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
archive_path = outputs / 'TokCom_stage49_frozen_holdout_continuation.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items:
        archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in manifest.items())
print(json.dumps(result, indent=2))
