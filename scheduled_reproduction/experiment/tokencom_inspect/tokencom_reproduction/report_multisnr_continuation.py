"""Render an audited continuation without overwriting historical session reports."""
import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
run = root / 'runs'
outputs = root.parents[2] / 'outputs'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
session_path = (root / args.session).resolve()
assert session_path.parent == run.resolve()
session = json.loads(session_path.read_text(encoding='utf-8'))
stage = session['stage']
assert isinstance(stage, int) and stage >= 55
plan = read('stage54_development_multisnr_plan.json')
phy = read('stage54_development_phy_audit.json')
model = read('stage54_development_recovery_audit.json')
checks = read('stage54_audit_contract_verification.json')
assert checks['passed'] == 4 and checks['failed'] == 0
assert checks['audit_script_sha256'] == sha(root / 'audit_stage54_development_recovery.py')
assert checks['check_script_sha256'] == sha(root / 'test_stage54_audit_contract.py')
assert phy['all_checks_passed'] and model['all_checks_passed']
assert phy['plan_sha256'] == model['plan_sha256'] == session['plan_sha256'] == sha(run / 'stage54_development_multisnr_plan.json')
assert session['primary_protocol_sha256'] == sha(run / 'stage44_frozen_protocol.json')
assert session['primary_summary_sha256'] == model['primary_summary_sha256'] == sha(run / 'stage47_complete_holdout_summary.json')
assert sha(root / 'run_stage46_frozen_recovery.py') == read('stage46_recovery_lock.json')['runner_sha256']
assert sha(root / 'evaluate_stage47_frozen_metrics.py') == read('stage47_metrics_lock.json')['metric_script_sha256']
assert session['initial_phy_cases'] <= phy['completed_cases'] <= 300
assert session['initial_model_cases'] <= model['completed_cases'] <= phy['completed_cases']
now = datetime.now(timezone.utc)
date = now.astimezone(timezone(timedelta(hours=8))).date().isoformat()
prefix = f'TokCom_阶段{stage}多SNR续跑_{date}'
evidence_names = ['stage54_development_multisnr_plan.json', 'stage54_development_phy_progress.json',
                  'stage54_development_phy_audit.json', 'stage54_development_recovery_lock.json',
                  'stage54_development_recovery_progress.json', 'stage54_development_recovery_audit.json',
                  'stage54_audit_contract_verification.json']
model_images = {str(snr): len({name.split('_')[1] for name in model['case_sha256']
                             if name.startswith(f'snr{si}_')}) for si, snr in enumerate(plan['snr_db'])}
phy_images = {str(snr): len({case['image_index'] for case in phy['cases'] if case['snr_index'] == si})
              for si, snr in enumerate(plan['snr_db'])}
result = dict(session, recorded_utc=now.isoformat(),
              phy_cases=phy['completed_cases'], new_phy_cases=phy['completed_cases']-session['initial_phy_cases'],
              paired_model_cases=model['completed_cases'], new_paired_model_cases=model['completed_cases']-session['initial_model_cases'],
              remaining_phy_cases=300-phy['completed_cases'], remaining_model_cases=300-model['completed_cases'],
              phy_counts_by_snr=phy['counts_by_snr'], model_counts_by_snr=model['counts_by_snr'],
              phy_images_by_snr=phy_images, model_images_by_snr=model_images,
              complete_phy_snr_groups=len(phy['complete_snr_groups']),
              complete_model_snr_groups=len(model['complete_snr_groups']), primary_holdout_changed=False,
              primary_recovery_cases=500, primary_metric_cases=500, new_primary_cases=0,
              primary_frozen_scripts_unchanged=True, audit_contract_checks_verified=4,
              full_paper_reproduction_complete=False)

# Preserve a session-named audit copy: the stage54 pilot audit is a resumable live view.
frozen_phy_name = f'stage{stage}_development_phy_audit.json'
frozen_model_name = f'stage{stage}_development_recovery_audit.json'
for source, target in [('stage54_development_phy_audit.json', frozen_phy_name),
                       ('stage54_development_recovery_audit.json', frozen_model_name)]:
    (run / target).write_bytes((run / source).read_bytes())
result['session_phy_audit'] = frozen_phy_name
result['session_model_audit'] = frozen_model_name

coverage = ['| SNR/dB | 物理接收/50 | 物理图像/10 | 四路径模型结果/50 | 模型图像/10 | 模型质量可汇总 |',
            '|---|---:|---:|---:|---:|---|']
physical = ['| SNR/dB | 真值数据包 PER | CRC 失败率 | 接收 token 错误率 | 单次 payload TCE | 理想 ARQ payload TCE（解析） |',
            '|---|---:|---:|---:|---:|---:|']
for snr in plan['snr_db']:
    key = str(snr)
    coverage.append(f"| {snr:g} | {phy['counts_by_snr'][key]} | {phy_images[key]} | {model['counts_by_snr'][key]} | {model_images[key]} | {'是' if key in model['complete_snr_groups'] else '否'} |")
    if key in phy['complete_snr_groups']:
        group = phy['complete_snr_groups'][key]
        per = group['true_packet_error_rate']
        assert math.isclose(group['ideal_arq_payload_tce'], 25.6*(1-per))
        physical.append(f"| {snr:g} | {per:.6%} | {group['crc_failure_rate']:.6%} | {group['received_token_error_rate']:.6%} | {group['single_shot_payload_tce']:.6f} | {group['ideal_arq_payload_tce']:.6f} |")
physical_text = chr(10).join(physical) if len(physical)>2 else '尚无完整的 50 接收物理组，不汇总部分组误差均值。'
exposure_text = ''
exposure_name = 'stage55_complete_phy_mask_exposure.json'
if (run / exposure_name).exists():
    exposure = read(exposure_name)
    assert exposure['all_checks_passed'] and exposure['plan_sha256'] == session['plan_sha256']
    assert exposure['phy_audit_sha256'] == sha(run / 'stage54_development_phy_audit.json')
    assert exposure['script_sha256'] == sha(root / 'summarize_stage55_complete_phy.py')
    evidence_names.append(exposure_name)
    rows = ['| SNR/dB | 真值错误包/3200 | CRC失败包/3200 | CRC失败但payload正确包 | CRC通过但payload错误包 | oracle掩码内原正确token | CRC掩码内原正确token |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for snr in plan['snr_db']:
        group = exposure['groups'][str(snr)]
        rows.append(f"| {snr:g} | {group['true_error_packets']} | {group['crc_failed_packets']} | {group['crc_failed_data_correct_packets']} | {group['crc_passed_data_wrong_packets']} | {group['oracle_masked_originally_correct_tokens']} | {group['crc_masked_originally_correct_tokens']} |")
    exposure_text = '\n## CRC 与 payload 错误定位及掩码覆盖\n\n'+chr(10).join(rows)+'\n\n每个 SNR 共 12800 个 token。两列掩码内原正确 token 统计的是生成前可能被替换的位置，不是生成后损坏数量。本试点两条 CMI 路径使用 oracle 掩码；CRC 掩码列仅描述物理接收数据。CRC 失败但 payload 正确的包不能与真值数据错误包等同；观测到零个 CRC 通过但 payload 错误包，也不证明总体漏检概率为零。\n'
figure_section = ''
convention_section = ''
convention_name = 'stage55_input_convention_audit.json'
if (run / convention_name).exists():
    convention = read(convention_name)
    assert convention['all_checks_passed']
    assert convention['script_sha256'] == sha(root / 'audit_stage55_input_conventions.py')
    assert all(value['remote_connector_identity_matches'] for value in convention['author_files'].values())
    evidence_names.append(convention_name)
    convention_section = '''## 作者训练与演示预处理差异

本轮对固定作者提交的四个文件完成 AST/文本及 GitHub 整文件 blob 身份核对。[训练代码 Trainer/vit.py 第183行](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/Trainer/vit.py#L183) 在编码前执行 2×x−1，把 [0,1] 输入映射到 [-1,1]；[演示 notebook](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/TokCom_Demo.ipynb) 的第8、14个单元则经 Resize/ToTensor 后直接编码 [0,1]。这确认了发布代码路径的差异，尚未证明公开权重实际使用哪套训练流水线，也未证明该差异导致本实验的质量表现。

另有指标口径差异：[作者 LPIPS 包装](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/Performance_metrics.py#L34) 未显式传 normalize 参数，而当前冻结指标显式使用 normalize=True。VQGAN decode_code 直接返回 decoder 输出；本轮没有更改编码输入、输出映射或指标约定。后续需预先定义独立旧开发图 round-trip 对照，固定图像几何和指标，再比较演示输入/原输出与训练范围输入/显式输出映射；不得重写主实验或把口径差异当作已证实因果。

'''
    range_plan_name = 'stage55_range_diagnostic_plan.json'
    range_plan = read(range_plan_name)
    assert range_plan['source_audit_sha256'] == sha(run / convention_name)
    assert range_plan['status'] == 'planned_not_executed' and len(range_plan['conditions']) == 4
    evidence_names.append(range_plan_name)
    convention_section += '独立对照计划已预先保存为 stage55_range_diagnostic_plan.json：固定十张旧开发图、图像几何和指标，比较输入 x / 2x−1 与输出 y / (y+1)/2 的四种组合，并报告全部条件和 token ID 差异。阶段55仅保存计划，不包含这项模型对照结果；后续是否执行应查看独立执行回执。计划不使用信道或 MaskGIT，不选择最佳条件，不改变当前多 SNR 试点。\n\n'
figure_paths = []
if len(phy['complete_snr_groups']) == len(plan['snr_db']):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = plan['snr_db']
    groups = [phy['complete_snr_groups'][str(snr)] for snr in x]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for key, label, marker in [('true_packet_error_rate', 'True data-packet PER', 'o'),
                               ('crc_failure_rate', 'CRC failure rate', 's'),
                               ('received_token_error_rate', 'Received token error rate', '^')]:
        axes[0].plot(x, [100*g[key] for g in groups], marker=marker, label=label)
    axes[0].set(xlabel='SNR (dB)', ylabel='Error / failure rate (%)', title='Measured single-shot PHY')
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [g['single_shot_payload_tce'] for g in groups], marker='o', label='Single-shot payload TCE')
    axes[1].plot(x, [g['ideal_arq_payload_tce'] for g in groups], marker='s', linestyle='--', label='Ideal ARQ: analytic, no retries simulated')
    axes[1].set(xlabel='SNR (dB)', ylabel='Pixels / token payload bit', title='Payload-only reference')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.grid(alpha=.25)
    fig.suptitle('Development pilot: 10 old images x 5 receptions per SNR\nAuthor-demo fixed interleaver; not paper-wide validation', fontsize=11)
    fig.supxlabel('CRC, coding, CMI, feedback and latency costs are excluded from payload TCE.', fontsize=9)
    for suffix in ('svg', 'png'):
        path = outputs / f'{prefix}_PHY.{suffix}'
        fig.savefig(path, dpi=160)
        figure_paths.append(path)
    plt.close(fig)
    result['phy_figure_sha256'] = {p.name: sha(p) for p in figure_paths}
    figure_section = f'\n![完整物理接收统计与解析 payload TCE]({prefix}_PHY.svg)\n\n图中实线物理误差来自完整缓存，理想 ARQ 虚线为解析计算；没有实际重传实验，也没有模型质量曲线。\n'
progress_name = f'stage{stage}_session_progress.json'
(run / progress_name).write_text(json.dumps(result, indent=2), encoding='utf-8')
text = f'''# TokCom 阶段{stage}：多 SNR 固定计划续跑

记录时间：{now.isoformat()}（UTC；本地 Asia/Shanghai）。本轮从物理 {session['initial_phy_cases']}/300、模型 {session['initial_model_cases']}/300 组继续。

## 实际增量与覆盖

本轮新增物理 {result['new_phy_cases']} 组、四路径配对恢复与指标 {result['new_paired_model_cases']} 组；当前累计物理 {phy['completed_cases']}/300、模型 {model['completed_cases']}/300。保存组全部通过输入哈希、最终 token、未掩码位置保持与有限指标检查。四路径为直接解码、无 CMI、有 CMI、正确 token 理想参考。

{chr(10).join(coverage)}

模型每个 SNR 需 10 图 × 5 次接收完整后才汇总质量；先对每图五次接收平均，再平均十图。当前完整模型组 {len(model['complete_snr_groups'])}/6；未完成组不输出质量均值。物理组和模型组分别判断完整性。

## 完整物理组与解析参考

{physical_text}
{figure_section}
TCE 单位为 pixels/token payload bit，单次为 65536/2560=25.6。理想 ARQ 使用真值包错误率 PER 的解析公式 25.6×(1−PER)，假设独立几何重试和理想识错；此处未实际运行重传。CRC 失败率与数据包 PER 分开统计，不将二者互换。每帧 payload 2560 bit、CRC 704 bit、卷积编码后 6528 bit、16-QAM 全帧 1632 符号；payload TCE 不计 CRC、编码、CMI、反馈、包头或延迟成本。类别标签直接提供给接收模型，尚未模拟其传输、误码与资源成本。

{exposure_text}

该试点使用旧十张开发图、作者演示固定交织、AWGN/CRC11/卷积码软 Viterbi/16-QAM。同一图与接收序号跨 SNR 重置相同噪声种子。CMI 两路径共享发送端真值 oracle 掩码，分别 w=0、w=1，生成参数、种子及执行锁保持不变。oracle 需要发送端真值，理想正确 token 解码也不是实际 ARQ 接收结果。

## 主实验与复现范围

冻结 8 dB 主实验维持 100 图 × 5 次接收、六路径，恢复与指标均为 500/500；本轮新增主实验组数为零。协议、完整汇总及两个执行脚本均匹配原冻结身份。

- 主协议 SHA-256：`{session['primary_protocol_sha256']}`。
- 主汇总 SHA-256：`{session['primary_summary_sha256']}`。
- 多 SNR 开发计划 SHA-256：`{session['plan_sha256']}`。

实际残余 TER 由最终 token ID 与发送 token 比较；作者 TER_af 返回值不作为残余 TER。图像指标沿用原始输出 PSNR、clamp 后 normalize=True 的 LPIPS、clamp→PIL→官方预处理的 CLIP。项目图像去重不证明与公开预训练数据独立。旧开发集结果、固定交织和解析参考不构成整篇论文复现；更大样本 CMI、资源等价基线及论文设置对齐仍未完成。

{convention_section}
## 发布与续跑

本报告基于发布前审计；发布提交、远端 CI 和精确 Git tree/commit 验证写入单独本地回执，避免自引用。持续更新草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。原始图像、权重和逐接收缓存不进入发布白名单。

剩余物理 {result['remaining_phy_cases']} 组、模型 {result['remaining_model_cases']} 组。保持原计划和执行锁，通过 cache_stage54_development_phy.py、audit_stage54_development_phy.py、run_stage54_development_multisnr.py、audit_stage54_development_recovery.py 续跑；已保存完整组自动跳过。历史阶段报告不重写，当前会话另存审计快照。
'''
report = outputs / f'{prefix}.md'
report.write_text(text, encoding='utf-8')
scripts = ['cache_stage54_development_phy.py', 'audit_stage54_development_phy.py',
           'run_stage54_development_multisnr.py', 'audit_stage54_development_recovery.py',
           'test_stage54_audit_contract.py', 'sync_github_worktree_from_api.py', 'report_multisnr_continuation.py']
if exposure_text:
    scripts.append('summarize_stage55_complete_phy.py')
if convention_section:
    scripts.append('audit_stage55_input_conventions.py')
evidence_names += [session_path.name, progress_name, frozen_phy_name, frozen_model_name]
entries = {f'code/{name}': (root/name).read_bytes() for name in scripts}
entries.update({f'evidence/{name}': (run/name).read_bytes() for name in evidence_names})
entries[f'reports/{report.name}'] = report.read_bytes()
entries.update({f'reports/{path.name}': path.read_bytes() for path in figure_paths})
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
entries['SHA256.json'] = json.dumps(manifest, indent=2).encode('utf-8')
archive = outputs / f'{prefix}_代码与证据.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name, data in entries.items():
        bundle.writestr(name, data)
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name, expected in manifest.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
print(json.dumps({'stage': stage, 'phy_cases': phy['completed_cases'], 'model_cases': model['completed_cases'],
                  'new_phy': result['new_phy_cases'], 'new_model': result['new_paired_model_cases'],
                  'archive_entries': len(entries), 'archive_sha256': sha(archive),
                  'report': str(report), 'archive': str(archive)}, ensure_ascii=True, indent=2))
