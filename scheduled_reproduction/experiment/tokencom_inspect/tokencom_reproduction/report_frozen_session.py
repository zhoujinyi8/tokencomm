"""Render audited continuation coverage from a saved session-start receipt."""
import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session', required=True, help='Session-start JSON relative to experiment root')
args = parser.parse_args()
root = Path(__file__).resolve().parent
run = root / 'runs'
outputs = root.parents[2] / 'outputs'
session_path = (root / args.session).resolve()
assert session_path.parent == run.resolve()
session = json.loads(session_path.read_text(encoding='utf-8'))
stage = int(session['stage'])
progress = json.loads((run / 'stage46_recovery_progress.json').read_text(encoding='utf-8'))
recovery_audit = json.loads((run / 'stage46_output_audit.json').read_text(encoding='utf-8'))
metrics = json.loads((run / 'stage47_metric_file_audit.json').read_text(encoding='utf-8'))
current = progress['completed_recovery_cases']
metric_count = metrics['metric_receptions_saved']
assert recovery_audit['all_checks_passed'] and recovery_audit['checked_cases'] == current
assert metrics['all_checks_passed'] and progress['metric_receptions_saved'] == metric_count
assert session['initial_recovery_cases'] <= current <= 500
assert session['initial_metric_cases'] <= metric_count <= current
summary_path = run / 'stage47_complete_holdout_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
if summary is not None:
    assert current == metric_count == 500
    assert summary['complete_images'] == 100 and summary['complete_receptions'] == 500
    assert summary['selected_exploratory_fraction'] is None
    assert len(summary['per_image']) == 100
    assert summary['protocol_sha256'] == hashlib.sha256((run / 'stage44_frozen_protocol.json').read_bytes()).hexdigest()
    assert summary['metrics_lock_sha256'] == hashlib.sha256((run / 'stage47_metrics_lock.json').read_bytes()).hexdigest()
elif metric_count == 500:
    raise ValueError('All metrics are available: run the frozen complete-summary entry point before reporting')
now = datetime.now(timezone.utc)
local_date = now.astimezone(timezone(timedelta(hours=8))).date().isoformat()
result = dict(session, recorded_utc=now.isoformat(), report_timezone='Asia/Shanghai',
              recovery_cases=current, new_recovery_cases=current-session['initial_recovery_cases'],
              complete_five_trial_images=progress['complete_five_trial_images'],
              metric_cases=metric_count, new_metric_cases=metric_count-session['initial_metric_cases'],
              complete_five_trial_metric_images=metrics['complete_five_trial_images'],
              recovery_complete=current == 500, metrics_complete=metric_count == 500,
              recovery_change_since_publication=current-session['previous_published_recovery_cases'],
              metric_change_since_publication=metric_count-session['previous_published_metric_cases'],
              expected_cases=500, paths_per_metric_case=6,
              frozen_scientific_settings_changed=False, model_performance_aggregated=summary is not None)
analysis_section = '当前仍未汇总部分样本性能。完整汇总入口要求 100 张图像、每张五轮信道和全部六条路径齐全，先按图像平均五轮，再汇总 100 图。恢复缓存完成本身不等于图像指标或完整论文复现完成。'
next_section = f'剩余恢复 {500-current} 组、指标 {500-metric_count} 组。继续按冻结设置补齐缺项，不依据留出结果选择门控策略。'
if summary is not None:
    result['complete_summary_sha256'] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    paths = ['direct', 'oracle', 'crc', 'soft_gate_0.05', 'soft_gate_0.1', 'soft_gate_0.25']
    labels = ['直接解码', '真值掩码恢复（oracle）', '接收端 CRC 恢复', '探索门控 5%', '探索门控 10%', '探索门控 25%']
    fields = ['correct_tokens', 'psnr_raw', 'lpips_normalized', 'clip_clipped']
    assert set(summary['means_over_images']) == set(paths)
    table = ['| 路径 | 正确 token 数/256 ↑ | PSNR/dB ↑ | LPIPS ↓ | CLIP ↑ |',
             '|---|---:|---:|---:|---:|']
    differences = ['| 路径减直接解码 | token 差值 | PSNR 差值/dB | LPIPS 差值 | CLIP 差值 |',
                   '|---|---:|---:|---:|---:|']
    for name, label in zip(paths, labels):
        values = [summary['means_over_images'][name][key] for key in fields]
        assert all(math.isfinite(value) for value in values)
        table.append(f'| {label} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.6f} | {values[3]:.6f} |')
        if name != 'direct':
            delta = [summary['paired_mean_differences_from_direct'][name][key] for key in fields]
            assert all(math.isfinite(value) for value in delta)
            differences.append(f'| {label} | {delta[0]:+.4f} | {delta[1]:+.4f} | {delta[2]:+.6f} | {delta[3]:+.6f} |')
    analysis_section = ('全部 500 组指标通过审计后，使用既有冻结入口生成全量汇总。每张图先平均五轮，再对 100 张图平均；三档探索门控全部报告，未选取某一档。以下是本协议下的描述性均值，不作显著性断言。\n\n'
                        + '\n'.join(table) + '\n\n配对差值同样先按图像计算，再平均 100 图；LPIPS 差值为负表示下降。\n\n'
                        + '\n'.join(differences)
                        + '\n\n真值掩码路径使用发送端真值，不能视作仅凭接收端信息取得的结果。PSNR 使用原始解码输出；LPIPS 使用裁剪输出和 normalize=True；CLIP 使用裁剪后转 PIL 的官方预处理，口径沿用冻结协议。')
    next_section = '本协议的恢复、图像指标和六路径配对汇总已经完成。后续转向论文 TCE、重传对照、CMI 消融及多 SNR 差距核对；工程反馈控制协议作为独立扩展评估，不使用该留出集继续调参或选择探索门控档位。'
progress_path = run / f'stage{stage}_session_progress.json'
progress_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
report = outputs / f'TokCom_阶段{stage}冻结留出续跑_{local_date}.md'
report.write_text(f'''# TokCom 阶段 {stage}：冻结留出续跑

报告日期 {local_date}（Asia/Shanghai）。本轮起点是 {session['initial_recovery_cases']} 组恢复、{session['initial_metric_cases']} 组指标；此前尚未发布的 {session.get('prior_unpublished_recovery_cases',0)} 组恢复已单独保存在前一阶段报告，不计入本轮新增。

恢复累计 {current}/500 组，本轮新增 {result['new_recovery_cases']} 组，五轮信道齐全图像 {progress['complete_five_trial_images']}/100 张。图像指标累计 {metric_count}/500 组，本轮新增 {result['new_metric_cases']} 组，五轮指标齐全图像 {metrics['complete_five_trial_images']}/100 张。六条路径合计 {metric_count*6} 条记录。相对 GitHub 上次发布，恢复共增加 {result['recovery_change_since_publication']} 组、指标增加 {result['metric_change_since_publication']} 组。

## 审计与执行

全部 {current} 组恢复通过执行锁、输入哈希、直接接收身份、未掩码位置、码本范围、门控构造及步数预算检查。全部 {metric_count} 组指标通过来源文件哈希、指标锁、六条路径及有限值审计。本轮使用独立隐藏后台进程运行原有冻结入口，未修改模型、协议、权重、种子、掩码、门控比例或指标定义。

{analysis_section}

## 发布与后续

继续更新同一草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。发布源码、冻结身份、报告和审计；原图、权重及生成缓存保留本地。实际发布提交、远程文件核验与 CI 结果记录在本轮回复和本地同步回执中。单元检查与依赖本地实验资产的集成检查分别说明，未运行的资产测试不计为通过。

{next_section}当前仅为 8 dB、固定生成种子、真实类别已知的项目留出评估；论文 TCE 曲线、重传对照、CMI 消融和多 SNR 复现尚未完成。工程反馈控制协议是另行报告的扩展。项目文件及像素去重不保证与公共预训练数据独立。
''', encoding='utf-8')
names = [Path(__file__).name, session_path.relative_to(root).as_posix(),
         progress_path.relative_to(root).as_posix(), 'runs/stage44_frozen_protocol.json',
         'runs/stage46_recovery_progress.json', 'runs/stage46_output_audit.json',
         'runs/stage46_recovery_lock.json', 'runs/stage47_metric_file_audit.json',
         'runs/stage47_metrics_lock.json']
if summary is not None:
    names.append(summary_path.relative_to(root).as_posix())
for name in session.get('supplemental_evidence', []):
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path.is_file()
    names.append(path.relative_to(root).as_posix())
items = [(root / name, name) for name in names] + [(report, report.name)]
for name in session.get('supplemental_reports', []):
    path = (outputs / name).resolve()
    assert path.parent == outputs.resolve() and path.is_file()
    items.append((path, path.name))
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in items}
archive_path = outputs / f'TokCom_stage{stage}_frozen_continuation.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path, name in items:
        archive.write(path, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in manifest.items())
print(json.dumps(result, indent=2))
print(report)
