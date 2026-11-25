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
if 'initial_model_case_sha256' in session:
    assert len(session['initial_model_case_sha256']) == session['initial_model_cases']
    for name, digest in session['initial_model_case_sha256'].items():
        assert sha(run/'stage54_development_recovery'/name) == digest == model['case_sha256'][name]
    assert sha(run/'stage54_development_recovery_lock.json') == session['model_execution_lock_sha256']
now = datetime.now(timezone.utc)
date = now.astimezone(timezone(timedelta(hours=8))).date().isoformat()
prefix = f'TokCom_阶段{stage}多SNR续跑_{date}'
evidence_names = ['stage54_development_multisnr_plan.json', 'stage54_development_phy_progress.json',
                  'stage54_development_phy_audit.json', 'stage54_development_recovery_lock.json',
                  'stage54_development_recovery_progress.json', 'stage54_development_recovery_audit.json',
                  'stage54_audit_contract_verification.json']
runtime_name = f'stage{stage}_runtime_identity.json'
runtime_section = ''
if (run/runtime_name).exists():
    runtime = read(runtime_name)
    assert runtime['all_checks_passed'] and runtime['packages_match_frozen_environment']
    assert runtime['author_sources_match_frozen_environment']
    assert runtime['baseline_environment_sha256'] == sha(run/'stage46_environment.json')
    assert runtime['runner_sha256'] == sha(root/'run_stage54_development_multisnr.py')
    assert runtime['class_mapping_matches_published_parent'] and runtime['class_mapping_matches_prior_lock']
    assert runtime['class_mapping_script_sha256'] == sha(root/'evaluate_stage9_pretrained_maskgit.py')
    evidence_names.append(runtime_name)
    result_runtime_verified = True
    runtime_section = '本轮另行核对Python版本、七个依赖包版本与十个作者源文件SHA-256，均与冻结环境记录一致；权重、源图和生成身份由实际模型执行入口核验。环境核对本身没有构造模型或运行推理。'
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
if runtime_section:
    result['runtime_identity_verified'] = result_runtime_verified
alignment_section = ''
next_study_section = ''
next_plan_name = f'stage{stage}_randomized_packetization_plan.json'
if (run/next_plan_name).exists():
    next_plan = read(next_plan_name)
    assert next_plan['status'] == 'planned_not_executed'
    assert next_plan['baseline_plan_sha256'] == session['plan_sha256']
    assert next_plan['baseline_runner_sha256'] == sha(root/'run_stage54_development_multisnr.py')
    assert next_plan['new_phy_simulations_performed'] == next_plan['new_model_inference_performed'] == 0
    assert len(next_plan['permutations']) == 10
    for row in next_plan['permutations']:
        permutation = row['permutation']; inverse = row['inverse_permutation']
        assert sorted(permutation) == list(range(256))
        assert all(permutation[inverse[i]] == i and inverse[permutation[i]] == i for i in range(256))
    evidence_names.append(next_plan_name)
    result['next_independent_plan'] = next_plan_name
    result['next_independent_plan_sha256'] = sha(run/next_plan_name)
    next_study_section = '''## 后续独立随机交织计划（尚未执行）

在当前六SNR质量汇总之前预先保存随机包分配敏感性计划：仍用同十张开发图、六SNR、每图五次接收，计划另算300组物理接收与四路径模型结果。每图使用固定PCG64种子202611260+图序号生成256位置置换，同图五次接收和六SNR共享该置换；接收ID与掩码经逆置换回到原位置。十个置换的双射与往返共40项检查通过，没有运行新物理模拟或模型推理。

仅改变包位置分配，保留原权重、输入输出约定、全帧QAM映射、噪声和生成种子、类别模式与指标；不同时引入补位或重传协议。固定与随机分支各SNR均齐备50次接收后，才汇总全部四路径及逐图配对差值。该计划不更改任何已有缓存，不是新留出集，也不声称恢复了论文未公开的具体置换种子。新执行锁须在首次推理前固定完整基线审计身份。
'''
alignment_name = f'stage{stage}_paper_alignment_review.json'
if (run/alignment_name).exists():
    alignment = read(alignment_name)
    assert alignment['primary_source_retrieved'] and not alignment['primary_or_multisnr_settings_changed']
    assert alignment['primary_protocol_sha256'] == session['primary_protocol_sha256']
    assert alignment['multisnr_plan_sha256'] == session['plan_sha256']
    assert alignment['coding_and_mapping']['phy_script_sha256'] == sha(root/'smoke_stage25_sionna_link.py')
    evidence_names.append(alignment_name)
    result['paper_alignment_review'] = alignment_name
    alignment_section = '''## 与论文实验范围的剩余差异

[论文VI-A/B](https://arxiv.org/html/2502.12096v5#S6)使用ImageNet100、随机化包内token位置及独立信道完美恢复的7-bit类别标签。本地开发集为十张Imagenette图像，冻结主实验为同来源十类的100张图；100张图不等同ImageNet100数据集。当前采用演示固定交织并直接提供类别条件，没有CMI传输资源实验。论文确切数据子集与划分仍需明确。

本地代码与Sionna源文件核对表明：CRC和卷积编码按包处理，输出每包102个编码bit；16-QAM阶段才把64包编码bit拼接映射为1632个符号。102不是4的整数倍，因此这个全帧符号数不能直接除以64后作为独立包重传的整数符号成本。若另行设计每包补2bit的独立映射，则每包26个符号、64包共1664个符号；这只是算术示例，未采用、未模拟，也不修改原接收或payload TCE结果。

后续实际ARQ须先固定包/符号边界、停止规则、重试噪声与类别资源核算，再实现相同条件的基线；当前理想正确token解码及解析TCE不替代物理重传实验。Cheng约0.04bpp基线仍未完成。
'''

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

阶段55对固定作者提交的四个文件完成 AST/文本及 GitHub 整文件 blob 身份核对。[训练代码 Trainer/vit.py 第183行](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/Trainer/vit.py#L183) 在编码前执行 2×x−1，把 [0,1] 输入映射到 [-1,1]；[演示 notebook](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/TokCom_Demo.ipynb) 的第8、14个单元则经 Resize/ToTensor 后直接编码 [0,1]。这确认了发布代码路径的差异，尚未证明公开权重实际使用哪套训练流水线，也未证明该差异导致本实验的质量表现。

另有指标口径差异：[作者 LPIPS 包装](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/Performance_metrics.py#L34) 未显式传 normalize 参数，而当前冻结指标显式使用 normalize=True。VQGAN decode_code 直接返回 decoder 输出；本轮没有更改编码输入、输出映射或指标约定。独立旧开发图 round-trip 与链路对照的执行状态见后节；不得重写主实验或把口径差异当作已证实因果。

'''
    range_plan_name = 'stage55_range_diagnostic_plan.json'
    range_plan = read(range_plan_name)
    assert range_plan['source_audit_sha256'] == sha(run / convention_name)
    assert range_plan['status'] == 'planned_not_executed' and len(range_plan['conditions']) == 4
    evidence_names.append(range_plan_name)
    convention_section += '独立对照计划已预先保存为 stage55_range_diagnostic_plan.json：固定十张旧开发图、图像几何和指标，比较输入 x / 2x−1 与输出 y / (y+1)/2 的四种组合，并报告全部条件和 token ID 差异。阶段55仅保存计划，不包含这项模型对照结果；后续是否执行应查看独立执行回执。计划不使用信道或 MaskGIT，不选择最佳条件，不改变当前多 SNR 试点。\n\n'
figure_paths = []
diagnostic_section = ''
diagnostic_scripts = []
if (run/'stage56_range_audit.json').exists():
    diagnostic = read('stage56_range_audit.json')
    assert diagnostic['all_checks_passed']
    assert diagnostic['plan_sha256'] == sha(run/'stage55_range_diagnostic_plan.json')
    evidence_names += ['stage56_range_execution_lock.json', 'stage56_range_progress.json', 'stage56_range_audit.json']
    diagnostic_scripts += ['run_stage56_range_diagnostic.py', 'audit_stage56_range_diagnostic.py']
    result['range_diagnostic_complete'] = diagnostic['complete']
    result['range_diagnostic_images'] = diagnostic['complete_images']
    diagnostic_section += '## 已执行的输入/输出映射对照\n\n'
    if diagnostic['complete']:
        paired = []
        metric_keys = ['psnr_raw', 'lpips_normalized', 'clip_clipped']
        baseline = diagnostic['condition_paths'][0]
        for case in diagnostic['per_image']:
            by_path = {row['path']: row for row in case['rows']}
            paired.append({'image_index': case['image_index'], 'source_sha256': case['source_sha256'],
                           'differences_from_demo_raw': {name: {key: by_path[name][key]-by_path[baseline][key]
                               for key in metric_keys} for name in diagnostic['condition_paths']}})
        for name in diagnostic['condition_paths']:
            for key in metric_keys:
                assert math.isclose(sum(item['differences_from_demo_raw'][name][key] for item in paired)/10,
                                    diagnostic['difference_from_demo_raw'][name][key], abs_tol=1e-12)
        result['range_paired_per_image_differences'] = paired
        rows = ['| 编码输入 | 解码输出映射 | PSNR/dB | LPIPS | CLIP |', '|---|---|---:|---:|---:|']
        labels = [('x', 'y'), ('x', '(y+1)/2'), ('2x−1', 'y'), ('2x−1', '(y+1)/2')]
        for name, (encoder, decoder) in zip(diagnostic['condition_paths'], labels):
            values = diagnostic['means_over_images'][name]
            rows.append(f"| {encoder} | {decoder} | {values['psnr_raw']:.6f} | {values['lpips_normalized']:.6f} | {values['clip_clipped']:.6f} |")
        diagnostic_section += chr(10).join(rows)
        diagnostic_section += f"\n\n十张旧开发图、四条件全部齐备，按图像平均。每张图四条件相对演示原输出的配对差值保存在本阶段 session_progress.json，均值与完整审计逐项核对。两种编码共有 {diagnostic['changed_token_ids']}/{diagnostic['total_token_ids']} 个位置的 token ID 不同（{diagnostic['changed_token_fraction']:.4%}），这是编码身份差异，不是信道 TER。原始像素与源图逐像素核对，PSNR 从保存输出重算；LPIPS/CLIP 的输入约定及权重受执行锁约束。没有选择或覆盖任何条件，也没有执行信道或 MaskGIT 恢复。\n\n"
    else:
        diagnostic_section += f"目前仅 {diagnostic['complete_images']}/10 图，未汇总部分样本质量。\n\n"
if (run/'stage56_prior_probe_audit.json').exists():
    prior = read('stage56_prior_probe_audit.json')
    assert prior['all_checks_passed']
    evidence_names += ['stage56_prior_probe_plan.json', 'stage56_prior_probe_lock.json',
                       'stage56_prior_probe_progress.json', 'stage56_prior_probe_audit.json']
    diagnostic_scripts += ['run_stage56_prior_probe.py', 'audit_stage56_prior_probe.py']
    result['prior_probe_complete'] = prior['complete']
    result['prior_probe_images'] = prior['complete_images']
    diagnostic_section += '## 固定掩码 token 先验匹配检查\n\n'
    if prior['complete']:
        rows = ['| 编码输入 | 类别条件 | 平均交叉熵/(nat/token) | 掩码位置 top-1 准确率 |', '|---|---|---:|---:|']
        for name in prior['paths']:
            values = prior['means_over_images'][name]
            encoder = '2x−1' if name.startswith('training_') else 'x'
            mode = '无 CMI' if name.endswith('without_cmi') else '有 CMI'
            rows.append(f"| {encoder} | {mode} | {values['cross_entropy']:.6f} | {values['top1_accuracy']:.4%} |")
        diagnostic_section += chr(10).join(rows)+'\n\n'
    else:
        diagnostic_section += f"仅 {prior['complete_images']}/10 图，未汇总部分样本。\n\n"
    diagnostic_section += '本节交叉熵 CE 的单位为 nat/token，与上文通信效率 TCE（pixels/token payload bit）不是同一指标。每图固定随机选择 128/256 个掩码位置，四条件共享位置；十图按图像平均。使用作者完整 1025 类 logits 计算交叉熵，审计从保存 logits 独立重算交叉熵和 top-1 计数。类别模式使用相同真值标签，分别丢弃或保留条件。不同输入编码会改变目标 token 分布，因此跨编码交叉熵与准确率不能单独判定哪种预处理与权重训练更匹配。此处没有迭代生成、物理信道或图像质量评估；其结果不等同信道恢复收益，也不证明公开权重的实际训练流水线。\n\n'
if (run/'stage56_end_to_end_convention_plan.json').exists():
    comparison = read('stage56_end_to_end_convention_plan.json')
    assert comparison['source_multisnr_plan_sha256'] == session['plan_sha256']
    evidence_names.append('stage56_end_to_end_convention_plan.json')
    comparison_audit_name = 'stage58_session_comparison_audit.json'
    if 'comparison_frozen_audit_sha256' in session:
        matched = read(comparison_audit_name)
        assert sha(run/comparison_audit_name) == session['comparison_frozen_audit_sha256']
        assert matched['all_checks_passed'] and matched['complete'] and matched['complete_pairs'] == 50
        assert matched['pipeline_receptions'] == session['comparison_complete_cases'] == 100
        assert matched['plan_sha256'] == sha(run/'stage56_end_to_end_convention_plan.json')
        for name, digest in matched['case_sha256'].items():
            assert sha(run/'stage57_comparison_recovery'/name) == digest
        result['comparison_study_changed'] = False
        result['comparison_complete_cases'] = 100
        result['comparison_complete_pairs'] = 50
        evidence_names.append(comparison_audit_name)
        diagnostic_section += '## 已完成且保持独立的8 dB链路约定对照\n\n'
        diagnostic_section += 'stage56_end_to_end_convention_plan.json 对应的独立对照已在阶段58完成：同十张旧开发图、8 dB、每图五次接收、两套约定各50组，共100个完整四路径接收组、50对接收。本轮重新核验阶段58冻结审计与100个结果文件哈希，没有重算、改写或把这些组转计入原六SNR试点。原六SNR模型覆盖仅按其独立目录的实际结果统计。\n\n'
        diagnostic_section += '完整对照中，训练代码约定相对演示约定的图像指标有变化，但两套约定下恢复后的token正确率均低于直接接收；训练代码约定的恢复PSNR也低于该约定的直接解码。因此没有依据开发图结果选择新的主实验流水线。全部四路径及逐图配对差值保存在阶段58冻结审计，输入映射诊断与固定掩码先验探测均不代替这项链路结果。\n\n'
    else:
        diagnostic_section += '完整链路对照已预先记录在 stage56_end_to_end_convention_plan.json：同十图、8 dB、每图五次接收、相同物理链路和生成参数，比较两套一致的输入/输出约定及全部四路径。只有两套流水线各50组齐备才做跨流水线质量汇总；具体执行状态以独立执行审计为准。\n\n'
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
transition_section = ''
transition_name = f'stage{stage}_token_transitions.json'
transition_contract_name = f'stage{stage}_token_transition_contract.json'
if (run/transition_name).exists():
    transitions = read(transition_name); transition_contract = read(transition_contract_name)
    assert transitions['all_checks_passed'] and transitions['source_audit_sha256'] == sha(run/'stage54_development_recovery_audit.json')
    assert transitions['script_sha256'] == sha(root/'summarize_multisnr_token_transitions.py')
    assert transition_contract['passed'] == 5 and transition_contract['failed'] == 0
    assert transition_contract['script_sha256'] == transitions['script_sha256']
    assert transition_contract['test_script_sha256'] == sha(root/'test_multisnr_token_transitions.py')
    evidence_names += [transition_name, transition_contract_name]
    result['token_transition_contract_checks'] = 5
    result['token_transition_audit'] = transition_name
    transition_section = '## 修复与原正确token改错的分解\n\n'
    if transitions['complete']:
        assert model['complete'] and transitions['receptions_audited'] == 300 and transitions['path_rows_audited'] == 1200
        rows = ['| SNR/dB | 恢复模式 | 接收原错误/12800 | 原错修复 | 原正确改错 | 净正确数增量 | 最终错误/12800 |',
                '|---|---|---:|---:|---:|---:|---:|']
        for snr in plan['snr_db']:
            for mode in ('without_cmi','with_cmi'):
                v = transitions['groups'][str(snr)]['totals'][mode]
                assert v['net_correct_gain'] == v['repaired']-v['damaged'] == v['wrong_before']-v['wrong_after']
                rows.append(f"| {snr:g} | {mode} | {v['wrong_before']} | {v['repaired']} | {v['damaged']} | {v['net_correct_gain']} | {v['wrong_after']} |")
        transition_section += chr(10).join(rows)
        transition_section += '\n\n每个SNR均为完整50次接收、12800个token。逐组从保存ID重新分类并严格核对：正确数变化=原错修复−原正确改错；未掩码位置保持不变。JSON记录全部四路径及逐图五次接收均值；直接解码修复与改错均为零，理想正确token参考修复全部原错且不损坏原正确token。这是既有结果的描述性分解，没有据此调整掩码、种子或模型，也没有重新推理。\n'
    else:
        transition_section += '六SNR整体尚未完成，不汇总部分接收的修复或改错计数。\n'
quality_section = '尚无完整50接收的模型SNR组，不输出部分组质量均值或模型质量曲线。'
result['quality_aggregated_snr_db'] = [snr for snr in plan['snr_db'] if str(snr) in model['complete_snr_groups']]
if model['complete_snr_groups']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete_snr = result['quality_aggregated_snr_db']
    table = ['| SNR/dB | 路径 | 正确token/256 | 残余TER | PSNR/dB | LPIPS | CLIP |',
             '|---|---|---:|---:|---:|---:|---:|']
    for snr in complete_snr:
        group = model['complete_snr_groups'][str(snr)]
        assert group['complete_receptions'] == 50
        assert set(group['means_over_images']) == set(plan['paths'])
        for name in plan['paths']:
            v = group['means_over_images'][name]
            assert all(math.isfinite(v[key]) for key in ('correct_tokens','whole_image_ter','psnr_raw','lpips_normalized','clip_clipped'))
            assert math.isclose(v['whole_image_ter'], 1-v['correct_tokens']/256, abs_tol=1e-12)
            table.append(f"| {snr:g} | {name} | {v['correct_tokens']:.6f} | {v['whole_image_ter']:.6%} | {v['psnr_raw']:.6f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    labels = {'direct':'Direct decode', 'without_cmi':'Without CMI', 'with_cmi':'With CMI',
              'ideal_correct_tokens':'Ideal correct-token reference'}
    fields = [('whole_image_ter','Final token error rate (%)',100), ('psnr_raw','Raw-output PSNR (dB)',1),
              ('lpips_normalized','Normalized LPIPS',1), ('clip_clipped','CLIP cosine',1)]
    for axis, (field, ylabel, scale) in zip(axes.flat, fields):
        for name, marker in zip(plan['paths'], ('o','s','^','D')):
            values = [scale*model['complete_snr_groups'][str(snr)]['means_over_images'][name][field] for snr in complete_snr]
            axis.plot(complete_snr, values, marker=marker, label=labels[name])
        axis.set(xlabel='SNR (dB)', ylabel=ylabel, xticks=plan['snr_db'])
        axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Old-development model study: all four paths, complete SNR groups only\n'
                 '10 images x 5 receptions per shown SNR; oracle error masks; no pipeline selected', fontsize=11)
    fig.supxlabel('Average five receptions within image, then ten images. Ideal tokens do not simulate physical retransmission.', fontsize=9)
    quality_figures = []
    for suffix in ('svg','png'):
        path = outputs/f'{prefix}_模型质量.{suffix}'
        fig.savefig(path, dpi=160); quality_figures.append(path)
    plt.close(fig)
    figure_paths += quality_figures
    result['model_quality_figure_sha256'] = {p.name:sha(p) for p in quality_figures}
    quality_section = chr(10).join(table)+f'\n\n![完整SNR组四路径模型质量]({prefix}_模型质量.svg)\n\n每个显示SNR均为完整50次接收，先对每图五次接收平均，再平均十图。四路径全部保留；token相等与图像质量分别解释，不依据任一指标选择流水线。图中理想正确token参考不是实际重传基线。'
progress_name = f'stage{stage}_session_progress.json'
publication_pr = session.get('github_pull_request', 1)
publication_section = ''
contract_name = f'stage{stage}_publication_branch_contract.json'
if (run/contract_name).exists():
    contract = read(contract_name)
    assert contract['passed'] == 4 and contract['failed'] == 0
    assert contract['helper_script_sha256'] == sha(root/'sync_github_worktree_from_api.py')
    assert contract['test_script_sha256'] == sha(root/'test_publication_branch_contract.py')
    evidence_names.append(contract_name)
    result['publication_branch_contract_checks'] = 4
    publication_section = '发布同步工具现支持显式--branch参数，在暂存和移动引用前核对当前分支。四项隔离Git检查验证错误分支拒绝、替代分支精确同步、重复执行一致，以及树不匹配拒绝移动引用；检查没有使用网络或模型推理。'
(run / progress_name).write_text(json.dumps(result, indent=2), encoding='utf-8')
text = f'''# TokCom 阶段{stage}：多 SNR 固定计划续跑

记录时间：{now.isoformat()}（UTC；本地 Asia/Shanghai）。本轮从物理 {session['initial_phy_cases']}/300、模型 {session['initial_model_cases']}/300 组继续。

## 实际增量与覆盖

本轮新增物理 {result['new_phy_cases']} 组、四路径配对恢复与指标 {result['new_paired_model_cases']} 组；当前累计物理 {phy['completed_cases']}/300、模型 {model['completed_cases']}/300。保存组全部通过输入哈希、最终 token、未掩码位置保持与有限指标检查。四路径为直接解码、无 CMI、有 CMI、正确 token 理想参考。

{runtime_section}

{chr(10).join(coverage)}

模型每个 SNR 需 10 图 × 5 次接收完整后才汇总质量；先对每图五次接收平均，再平均十图。当前完整模型组 {len(model['complete_snr_groups'])}/6；未完成组不输出质量均值。物理组和模型组分别判断完整性。

## 完整模型组的四路径质量

{quality_section}

{transition_section}

## 完整物理组与解析参考

{physical_text}
{figure_section}
TCE 单位为 pixels/token payload bit，单次为 65536/2560=25.6。理想 ARQ 使用真值包错误率 PER 的解析公式 25.6×(1−PER)，假设独立几何重试和理想识错；此处未实际运行重传。CRC 失败率与数据包 PER 分开统计，不将二者互换。每帧 payload 2560 bit、CRC 704 bit、卷积编码后 6528 bit、16-QAM 全帧 1632 符号；payload TCE 不计 CRC、编码、CMI、反馈、包头或延迟成本。类别标签直接提供给接收模型，尚未模拟其传输、误码与资源成本。

{exposure_text}

该试点使用旧十张开发图、作者演示固定交织、AWGN/CRC11/卷积码软 Viterbi/16-QAM。同一图与接收序号跨 SNR 重置相同噪声种子。CMI 两路径共享发送端真值 oracle 掩码，分别 w=0、w=1，生成参数、种子及执行锁保持不变。oracle 需要发送端真值，理想正确 token 解码也不是实际 ARQ 接收结果。

本节完整物理统计与多 SNR 恢复均沿用演示输入约定；输入范围和 token 先验诊断使用独立缓存，不替换这些结果。

## 主实验与复现范围

冻结 8 dB 主实验维持 100 图 × 5 次接收、六路径，恢复与指标均为 500/500；本轮新增主实验组数为零。协议、完整汇总及两个执行脚本均匹配原冻结身份。

- 主协议 SHA-256：`{session['primary_protocol_sha256']}`。
- 主汇总 SHA-256：`{session['primary_summary_sha256']}`。
- 多 SNR 开发计划 SHA-256：`{session['plan_sha256']}`。

实际残余 TER 由最终 token ID 与发送 token 比较；作者 TER_af 返回值不作为残余 TER。图像指标沿用原始输出 PSNR、clamp 后 normalize=True 的 LPIPS、clamp→PIL→官方预处理的 CLIP。项目图像去重不证明与公开预训练数据独立。旧开发集结果、固定交织和解析参考不构成整篇论文复现；更大样本 CMI、资源等价基线及论文设置对齐仍未完成。

{alignment_section}

{convention_section}
{diagnostic_section}
{next_study_section}
## 发布与续跑

本报告基于发布前审计；发布提交、远端 CI 和精确 Git tree/commit 验证写入单独本地回执，避免自引用。持续更新草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/{publication_pr} 。原始图像、权重和逐接收缓存不进入发布白名单。

{publication_section}

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
scripts += diagnostic_scripts
if publication_section:
    scripts.append('test_publication_branch_contract.py')
if transition_section:
    scripts += ['summarize_multisnr_token_transitions.py','test_multisnr_token_transitions.py']
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
                  'report': str(report), 'archive': str(archive),
                  'figures': [str(path) for path in figure_paths]}, ensure_ascii=True, indent=2))
