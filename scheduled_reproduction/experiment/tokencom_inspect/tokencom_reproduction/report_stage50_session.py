"""Record recovery completion and metric continuation under existing locks."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
outputs = root.parents[2] / 'outputs'
progress = json.loads((run / 'stage46_recovery_progress.json').read_text(encoding='utf-8'))
recovery_audit = json.loads((run / 'stage46_output_audit.json').read_text(encoding='utf-8'))
metric_audit = json.loads((run / 'stage47_metric_file_audit.json').read_text(encoding='utf-8'))
prior_sync = json.loads((run / 'stage46_github_sync_receipt.json').read_text(encoding='utf-8'))
current = progress['completed_recovery_cases']
metric_count = metric_audit['metric_receptions_saved']
assert recovery_audit['all_checks_passed'] and recovery_audit['checked_cases'] == current
assert metric_audit['all_checks_passed'] and 35 <= metric_count <= current <= 500
assert 419 <= current
assert not (run / 'stage47_complete_holdout_summary.json').exists()
result = {
    'session_start_utc': '2026-11-22T14:57:30.437Z',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'initial_recovery_cases': 419,
    'initial_metric_cases': 35,
    'recovery_cases': current,
    'new_recovery_cases': current - 419,
    'recovery_complete': current == 500,
    'complete_five_trial_images': progress['complete_five_trial_images'],
    'metric_cases': metric_count,
    'new_metric_cases': metric_count - 35,
    'complete_five_trial_metric_images': metric_audit['complete_five_trial_images'],
    'expected_cases': 500,
    'paths_per_metric_case': 6,
    'model_performance_aggregated': False,
    'frozen_scientific_settings_changed': False,
    'prior_verified_ci': {
        key: prior_sync[key] for key in (
            'commit', 'ci_verified_run', 'latest_commit_ci',
            'unit_tests_passed', 'unit_tests_skipped', 'asset_integrations_deselected')
    },
    'current_publication_receipt': 'runs/stage46_github_sync_receipt.json',
}
(run / 'stage50_session_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
status = '恢复缓存已全部完成' if current == 500 else '恢复缓存继续补齐中'
report = outputs / 'TokCom_第五十阶段恢复覆盖与指标续跑_2026-11-22.md'
report.write_text(f'''# TokCom 第五十阶段：恢复覆盖与指标续跑

本轮起点为 419 组恢复、35 组图像指标。当前恢复累计 {current}/500 组，本轮新增 {current-419} 组；五轮信道均完成图像 {progress['complete_five_trial_images']}/100 张。{status}。这表示作者冻结路径的 token 恢复执行覆盖，不等于图像指标评估或完整论文复现已经完成。

逐组图像指标累计 {metric_count}/500 组，本轮新增 {metric_count-35} 组；五轮指标齐全图像 {metric_audit['complete_five_trial_images']}/100 张。每组包含直接解码、真值掩码恢复、CRC 恢复及三档软门控，共 {metric_count*6} 条路径记录。原有 100 个作者编码文件与 500 组物理接收文件已完成，后续无需重新采样或重算。

## 冻结身份与审计

已保存的全部 {current} 组恢复结果通过执行锁、输入哈希、直接接收身份、未掩码位置、码本范围、门控构造及步数预算审计。全部 {metric_count} 组指标通过来源恢复文件哈希、指标锁、六条路径和有限值审计。

本轮继续沿用锁定的作者模型、权重、种子、8 dB 条件及指标定义，没有因留出结果调参。当前没有汇总部分样本性能，也没有选择三档探索门控中的某一档。最终入口要求 100 张图像、每张五轮信道及六条路径全部齐全，再按图像平均五轮并汇总 100 图。

## GitHub 与验证范围

继续更新同一草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。发布代码、冻结清单、审计及报告；原图、模型权重和大体积生成缓存保留本地。

本轮开始时已验证提交 `{prior_sync['commit']}`，GitHub 检查运行 {prior_sync['ci_verified_run']} 成功：96 项单元测试通过、2 项跳过、11 项依赖本地实验资产的集成测试未运行。新发布提交、远程完整文件核验和新 CI 结果写入同步回执并在本轮回复报告；单元检查不代表资产集成或完整论文结果通过。

## 后续

剩余恢复 {500-current} 组，剩余指标 {500-metric_count} 组。继续补齐缺项，完成全量指标后再给出六条路径的图像配对汇总。当前仍为单一 SNR、固定生成种子、真实类别已知的项目留出评估；完整 TCE、反馈协议和多 SNR 复现尚未完成。项目图像文件及像素去重不保证与公共预训练数据独立。
''', encoding='utf-8')
names = [
    'report_stage50_session.py', 'run_stage46_frozen_recovery.py',
    'evaluate_stage47_frozen_metrics.py', 'summarize_stage47_complete_holdout.py',
    'audit_stage46_outputs.py', 'audit_stage47_metric_files.py',
    'runs/stage50_session_progress.json', 'runs/stage44_frozen_protocol.json',
    'runs/stage46_recovery_progress.json', 'runs/stage46_output_audit.json',
    'runs/stage46_recovery_lock.json', 'runs/stage47_metrics_lock.json',
    'runs/stage47_metric_file_audit.json',
]
items = [(name, root) for name in names] + [(report.name, outputs)]
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
archive_path = outputs / 'TokCom_stage50_recovery_coverage_metrics.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items:
        archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in manifest.items())
print(json.dumps(result, indent=2))
