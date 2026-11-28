"""Render the separate multi-SNR pilot from completed audits and package evidence."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'runs'
outputs = root.parents[2] / 'outputs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
session = read('stage54_window_start.json')
plan = read('stage54_development_multisnr_plan.json')
phy = read('stage54_development_phy_audit.json')
model = read('stage54_development_recovery_audit.json')
checks = read('stage54_audit_contract_verification.json')
assert checks['passed'] == 4 and checks['failed'] == 0
assert checks['audit_script_sha256'] == sha(root / 'audit_stage54_development_recovery.py')
assert checks['check_script_sha256'] == sha(root / 'test_stage54_audit_contract.py')
assert phy['all_checks_passed'] and model['all_checks_passed']
assert phy['plan_sha256'] == model['plan_sha256'] == sha(run / 'stage54_development_multisnr_plan.json')
assert session['primary_protocol_sha256'] == sha(run / 'stage44_frozen_protocol.json')
assert session['primary_summary_sha256'] == model['primary_summary_sha256'] == sha(run / 'stage47_complete_holdout_summary.json')
assert sha(root / 'run_stage46_frozen_recovery.py') == read('stage46_recovery_lock.json')['runner_sha256']
assert sha(root / 'evaluate_stage47_frozen_metrics.py') == read('stage47_metrics_lock.json')['metric_script_sha256']
assert model['completed_cases'] <= phy['completed_cases'] <= 300
model_images = {str(snr): len({name.split('_')[1] for name in model['case_sha256']
                             if name.startswith(f'snr{si}_')}) for si, snr in enumerate(plan['snr_db'])}
phy_images = {str(snr): len({case['image_index'] for case in phy['cases'] if case['snr_index'] == si})
              for si, snr in enumerate(plan['snr_db'])}
result = dict(session, recorded_utc=datetime.now(timezone.utc).isoformat(),
              phy_cases=phy['completed_cases'], new_phy_cases=phy['completed_cases'],
              paired_model_cases=model['completed_cases'], new_paired_model_cases=model['completed_cases'],
              remaining_phy_cases=300-phy['completed_cases'],
              remaining_model_cases=300-model['completed_cases'],
              phy_counts_by_snr=phy['counts_by_snr'], model_counts_by_snr=model['counts_by_snr'],
              phy_images_by_snr=phy_images, model_images_by_snr=model_images,
              complete_model_snr_groups=len(model['complete_snr_groups']),
              primary_holdout_changed=False, primary_recovery_cases=500, primary_metric_cases=500,
              new_primary_cases=0, primary_frozen_scripts_unchanged=True,
              audit_contract_checks_passed=4, full_paper_reproduction_complete=False)
(run / 'stage54_session_progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
table = ['| SNR/dB | 物理接收缓存/50 | 物理覆盖图像/10 | 四路径恢复与指标/50 | 模型覆盖图像/10 | 可汇总质量均值 |',
         '|---|---:|---:|---:|---:|---|']
for snr in plan['snr_db']:
    key = str(snr)
    table.append(f"| {snr:g} | {phy['counts_by_snr'][key]} | {phy_images[key]} | {model['counts_by_snr'][key]} | {model_images[key]} | {'是' if key in model['complete_snr_groups'] else '否，仅报告覆盖量'} |")
text = f'''# TokCom 阶段54：多 SNR 开发试点与 Git 同步恢复

记录时间：{result['recorded_utc']}（UTC；本地时区 Asia/Shanghai）。

## 本轮实际产出

独立多 SNR 开发试点新增并审计物理接收 {phy['completed_cases']}/300 组，四路径配对恢复及图像指标 {model['completed_cases']}/300 组。物理缓存、最终 token、未掩码位置保持、输入身份和指标有限性检查均通过。四条路径为直接解码、无 CMI、有 CMI、正确 token 理想参考；每个已保存接收组同时保存全部四条路径。

{chr(10).join(table)}

每个 SNR 预先要求 10 图 × 5 次接收。只有齐备的 50 组才计算该 SNR 的质量/误差均值；先对每图五次接收平均，再汇总十张图。目前完整模型 SNR 组数 {len(model['complete_snr_groups'])}/6。未完成组不输出性能均值，也不将缺项记作零。

## 固定设置与解释范围

使用旧开发集十张图，六个 SNR 为 6、6.5、7、7.5、8、9 dB；不是新的独立验证集。物理层沿用作者演示实现：AWGN、CRC11、[101,111] 卷积码、软 Viterbi、16-QAM，每包四个 token。固定交织位置来自作者演示，尚未等同论文随机包位置设置。同一图/接收序号在不同 SNR 使用相同随机种子以控制噪声差异。

CMI 两路径采用同一发送端真值 oracle 包掩码，w=0 去除类别条件，w=1 使用真值 ImageNet 类别；固定生成种子 20261124、32 步及原定温度/调度，不依本轮结果选参数。oracle 需要发送端真值，不能解释为接收端可直接部署的错误定位。实际误差用最终 token ID 与发送 token 比较；作者 GenDecoding 的 TER_af 返回值不是本报告的残余 TER。

每帧 256 个 10-bit token，即 payload 2560 bit；64 个 CRC11 共 704 bit，卷积码后 6528 bit，16-QAM 全帧映射为 1632 个符号。此账本不包含反馈、包头或额外重传。正确 token 理想解码是上限参考，未实际模拟物理重传；解析 ARQ/TCE 参考与实测质量分开。

图像指标沿用原定义：原始解码输出 PSNR、钳制图像且 normalize=True 的 LPIPS、钳制到 PIL 后官方 CLIP 预处理的余弦相似度。权重、计划、脚本及输入身份由锁和审计记录。

审计入口通过四项隔离数据契约检查：缺项组无均值、完整 50 组正确聚合、错误 token 计数被拒绝、未掩码位置改动被拒绝。检查不加载模型，不计入模型性能样本；它独立于远端仓库单元测试。

## 已完成主实验保持不变

100 图 × 5 次接收的冻结 8 dB 主实验仍为恢复 500/500、图像指标 500/500，每组六条路径。本轮新增主实验接收组为 0；未调整门控、协议、权重或主结果。

- 主协议 SHA-256：`{session['primary_protocol_sha256']}`。
- 主汇总 SHA-256：`{session['primary_summary_sha256']}`。
- 新开发试点计划 SHA-256：`{phy['plan_sha256']}`。

主实验已完成不代表整篇论文复现完成。跨 SNR、CMI 更大样本、资源等价基线和论文数据集/模型/指标口径对齐仍需继续；项目图像去重也未证明与公开预训练数据独立。

## GitHub 与可核验材料

此前本地工作树落后远端提交，已恢复到父提交 `{session['parent_commit']}`。恢复过程先匹配远端完整 Git tree，再重建并逐字节核对原提交对象 SHA，验证后才更新本地分支引用；记录见 stage54_local_sync_recovery.json 与幂等验证文件。新 helper 可在原生 fetch 不通时通过 GitHub API 元数据验证发布提交及完整文件树。

本报告记录发布前审计状态。最终发布提交、远端 CI 和发布后的本地同步结果单独写入本地 GitHub 同步回执，避免提交内自引用。持续更新地址：https://github.com/zhoujinyi8/tokencomm/pull/1 。原始图像、模型权重、逐接收生成缓存和账户信息不进入发布白名单。

## 后续续跑

剩余物理接收 {300-phy['completed_cases']} 组、配对恢复与指标 {300-model['completed_cases']} 组。先在 Sionna 环境运行 cache_stage54_development_phy.py 并重新审计，再在 PyTorch 环境运行 run_stage54_development_multisnr.py；程序在相同计划和脚本锁下跳过已保存完整组。完成窗口后运行 audit_stage54_development_recovery.py。每个 SNR 完整后才形成该组结果；不得用冻结主验证集选择参数。
'''
report = outputs / 'TokCom_阶段54多SNR开发试点与同步恢复_2026-11-24.md'
report.write_text(text, encoding='utf-8')
scripts = ['cache_stage54_development_phy.py', 'audit_stage54_development_phy.py',
           'run_stage54_development_multisnr.py', 'audit_stage54_development_recovery.py',
           'sync_github_worktree_from_api.py', 'report_stage54_session.py', 'test_stage54_audit_contract.py']
evidence = ['stage54_window_start.json', 'stage54_local_sync_recovery.json',
            'stage54_api_sync_idempotence.json', 'stage54_development_multisnr_plan.json',
            'stage54_development_phy_progress.json', 'stage54_development_phy_audit.json',
            'stage54_development_recovery_lock.json', 'stage54_development_recovery_progress.json',
            'stage54_development_recovery_audit.json', 'stage54_session_progress.json',
            'stage54_audit_contract_verification.json']
entries = {name: (root/name).read_bytes() for name in scripts}
entries.update({f'evidence/{name}': (run/name).read_bytes() for name in evidence})
entries[f'reports/{report.name}'] = report.read_bytes()
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
entries['SHA256.json'] = json.dumps(manifest, indent=2).encode('utf-8')
archive = outputs / 'TokCom_阶段54代码与审计证据_2026-11-24.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name, data in entries.items():
        bundle.writestr(name, data)
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name, digest in manifest.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
print(json.dumps({'phy_cases': phy['completed_cases'], 'model_cases': model['completed_cases'],
                  'complete_snr_groups': len(model['complete_snr_groups']),
                  'archive_entries': len(entries), 'archive_sha256': sha(archive),
                  'report': str(report), 'archive': str(archive)}, ensure_ascii=True, indent=2))
