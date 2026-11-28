"""Report a matched-convention session using audited coverage and explicit reuse counts."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent; run = root/'runs'; outputs = root.parents[2]/'outputs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: json.loads((run/n).read_text(encoding='utf-8'))
session_path = (root/args.session).resolve(); assert session_path.parent == run.resolve()
session = json.loads(session_path.read_text(encoding='utf-8')); stage = session['stage']
assert stage >= 57
audit = read('stage57_comparison_audit.json'); phy = read('stage57_comparison_phy_audit.json')
plan = read('stage56_end_to_end_convention_plan.json')
tests = read('stage57_aggregation_contract_verification.json')
assert audit['all_checks_passed'] and not audit.get('snapshot_only', False)
assert phy['all_checks_passed'] and phy['complete']
assert audit['plan_sha256'] == phy['plan_sha256'] == session['comparison_plan_sha256'] == sha(run/'stage56_end_to_end_convention_plan.json')
assert tests['passed'] == 4 and tests['failed'] == 0
assert tests['aggregation_script_sha256'] == sha(root/'comparison_aggregation.py')
assert tests['test_script_sha256'] == sha(root/'test_stage57_comparison_aggregation.py')
assert sha(run/'stage44_frozen_protocol.json') == session['primary_protocol_sha256']
assert sha(run/'stage47_complete_holdout_summary.json') == session['primary_summary_sha256']
old = read('stage54_development_recovery_audit.json')
assert old['completed_cases'] == session['initial_old_multisnr_cases']
now = datetime.now(timezone.utc); date = now.astimezone(timezone(timedelta(hours=8))).date().isoformat()
prefix = f'TokCom_阶段{stage}完整链路输入约定对照_{date}'
assert 0 <= session['initial_computed_cases'] <= audit['computed_cases']
assert 0 <= session['initial_reused_cases'] <= audit['reused_demo_cases']
assert session['initial_comparison_cases'] == session['initial_computed_cases']+session['initial_reused_cases']
result = dict(session, recorded_utc=now.isoformat(), comparison_counts=audit['counts'],
    comparison_pipeline_receptions=audit['pipeline_receptions'], complete_comparison_pairs=audit['complete_pairs'],
    comparison_reused_demo_cases=audit['reused_demo_cases'], comparison_computed_cases=audit['computed_cases'],
    new_computed_cases=audit['computed_cases']-session['initial_computed_cases'],
    newly_catalogued_reused_cases=audit['reused_demo_cases']-session['initial_reused_cases'],
    computed_cases_by_pipeline=audit['computed_cases_by_pipeline'],
    comparison_complete=audit['complete'], quality_means_aggregated=audit['quality_means_aggregated'],
    remaining_pipeline_receptions=100-audit['pipeline_receptions'],
    remaining_complete_pairs=50-audit['complete_pairs'], primary_holdout_changed=False,
    old_multisnr_model_cases=old['completed_cases'], old_multisnr_study_changed=False,
    full_paper_reproduction_complete=False)
audit_copy = f'stage{stage}_session_comparison_audit.json'; (run/audit_copy).write_bytes((run/'stage57_comparison_audit.json').read_bytes())
coverage = ['| 流水线 | 完整四路径接收组/50 | 复用旧恢复 | 本研究计算 | 图像覆盖/10 |', '|---|---:|---:|---:|---:|']
for pipeline in ('demo', 'training_code_convention'):
    coverage.append(f"| {pipeline} | {audit['counts'][pipeline]} | {audit['reused_demo_cases'] if pipeline == 'demo' else 0} | {audit['computed_cases_by_pipeline'][pipeline]} | {audit['images_by_pipeline'][pipeline]} |")
physical = ['| 流水线 | 物理接收组 | 真值错误包/3200 | 接收错误token/12800 | CRC失败包/3200 |', '|---|---:|---:|---:|---:|']
for pipeline in ('demo', 'training_code_convention'):
    values = {key: sum(case['pipelines'][pipeline][key] for case in phy['cases'])
              for key in ('true_error_packets', 'wrong_tokens', 'crc_failed_packets')}
    physical.append(f"| {pipeline} | 50 | {values['true_error_packets']} | {values['wrong_tokens']} | {values['crc_failed_packets']} |")
quality = '两套流水线尚未各齐备50组，不输出跨流水线图像质量均值或效果结论。'
figure_paths = []
if audit['complete']:
    table = ['| 流水线 | 路径 | 正确token/256 | PSNR/dB | LPIPS | CLIP |', '|---|---|---:|---:|---:|---:|']
    for pipeline in ('demo', 'training_code_convention'):
        for path in plan['paths']:
            v = audit['means_over_images'][pipeline][path]
            table.append(f"| {pipeline} | {path} | {v['correct_tokens']:.6f} | {v['psnr_raw']:.6f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
    quality = chr(10).join(table)+'\n\n先对每图五次接收平均，再汇总十图；全部逐图配对差值保存在完整审计 JSON，不选择最佳流水线。'
    differences = ['| 路径：训练代码约定减演示约定 | 正确token差值 | PSNR差值/dB | LPIPS差值 | CLIP差值 |',
                   '|---|---:|---:|---:|---:|']
    for path in plan['paths']:
        v = audit['paired_mean_difference'][path]
        differences.append(f"| {path} | {v['correct_tokens']:.6f} | {v['psnr_raw']:.6f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
    quality += '\n\n'+chr(10).join(differences)
    from plot_comparison_differences import plot_paired_differences
    figure_paths = plot_paired_differences(audit, outputs, prefix)
    result['paired_difference_figure_sha256'] = {p.name: sha(p) for p in figure_paths}
    quality += f'\n\n![逐图配对差值]({prefix}_配对差值.svg)\n\n每个点是同图五次接收平均后的差值，黑色菱形是十图配对均值。图不表示独立验证或显著性检验；PSNR/CLIP正值与LPIPS负值分别对应训练代码约定下该指标数值方向更优，不能据单指标选择流水线。'
progress_name = f'stage{stage}_session_progress.json'
comparison_interpretation = ('本对照已完整测量两套约定的信道恢复链路；结果只描述这十张开发图，不用于选择或改写主实验流水线。'
                             if audit['complete'] else '本对照正在测量完整信道链路，配对完成前不推断恢复收益。')
resume_text = ('本对照100个流水线接收组与50对接收均完成，四路径质量及逐图配对差值已汇总。后续可继续原六SNR开发研究剩余接收组；保留本对照独立结果与全部执行锁，不依据开发图结果选择新的主实验流水线。'
               if audit['complete'] else f"剩余 {100-audit['pipeline_receptions']} 个流水线接收组、{50-audit['complete_pairs']} 对完整接收。保持计划与执行锁运行 run_stage57_convention_comparison.py，随后执行 audit_stage57_convention_comparison.py；已保存组自动跳过。完整50对才汇总四路径质量与逐图差值。")
provenance = ''
if (run/'stage58_checkpoint_metadata_audit.json').exists():
    metadata = read('stage58_checkpoint_metadata_audit.json')
    assert metadata['weights_verified'] and metadata['script_sha256'] == sha(root/'audit_checkpoint_metadata.py')
    result['checkpoint_training_preprocessing_established'] = False
    provenance = '## 权重来源与元数据边界\n\n本轮以安全只读方式核验两份权重SHA-256。VQGAN旧格式只解析四段pickle文件头操作码，未反序列化对象或分配tensor；MaskGIT使用weights_only=True与内存映射读取显式元数据，没有构造网络或运行推理。已读取的元数据没有建立可追溯的训练预处理记录；旧VQGAN检查仅限文件头标记，不能据此断言元数据完全不存在。\n\n'
    if (run/'stage58_upstream_weight_source_audit.json').exists():
        source = read('stage58_upstream_weight_source_audit.json')
        if 'weight_files' in source:
            assert all(any(item['matches'] for item in items) for items in source['weight_files'].values())
            revision = source['repository_revision']
            training = source['published_upstream_training_source']
            provenance += f"两份本地权重均与[提供库固定版本](https://huggingface.co/llvictorll/Maskgit-pytorch/tree/{revision})中的对应LFS对象SHA-256一致。提供库revision：`{revision}`。[上游发布训练实现]({training['source_url']})在编码前同样执行2x−1；源文件Git blob：`{training['training_file_git_blob_sha']}`。这些证据确认二进制身份与已发布代码写法，仍不证明实际训练运行的全部预处理，也不依据性能结果反推训练口径。\n\n"
(run/progress_name).write_text(json.dumps(result, indent=2), encoding='utf-8')
report = outputs/f'{prefix}.md'
report.write_text(f'''# TokCom 阶段{stage}：完整链路输入约定对照

记录时间：{now.isoformat()}（UTC；本地 Asia/Shanghai）。执行预先保存的 stage56_end_to_end_convention_plan.json，独立于原冻结研究。

## 当前实际覆盖与本轮增量

{chr(10).join(coverage)}

当前共 {audit['pipeline_receptions']}/100 个完整流水线接收组，其中复用旧恢复 {audit['reused_demo_cases']} 组、本研究计算 {audit['computed_cases']} 组。本轮新计算 {result['new_computed_cases']} 组；同图同次接收的完整配对为 {audit['complete_pairs']}/50，五次接收均配对齐备的图像为 {audit['complete_five_trial_images']}/10。每个接收组包含直接解码、无 CMI、有 CMI、正确 token 理想参考四路径。复用与新计算分别记录，不把复用计作新模型推理。

## 两套流水线的完整物理接收

{chr(10).join(physical)}

本轮实际新增物理模拟 {session.get('new_phy_simulations', 0)} 组；演示约定50组来自已审计缓存。两套流水线各10图×5次接收，8 dB、相同图像/接收序号噪声种子、CRC11、[101,111]卷积码软Viterbi、16-QAM及作者演示固定交织。每帧256个10-bit token，payload 2560 bit，CRC 704 bit，编码后6528 bit，全帧1632个QAM符号。不同编码改变发送符号，因而错误包位置可以不同；每套流水线根据自身真值错误包生成oracle掩码，两种类别模式在同套流水线内共享掩码。

## 约定、身份和复用条件

- demo：原图x∈[0,1]直接编码，原始解码输出y参与指标。
- training_code_convention：编码2x−1，全部四路径的解码输出统一映射为(y+1)/2后参与指标。名称来自训练代码写法，不证明公开权重实际训练预处理。
- 两套均固定RGB、Resize((256,256))、相同VQGAN/MaskGIT权重，生成种子20261124、32步、sm_temp=1.3、r_temp=7、linear随机化与arccos调度。w=0/1分别去除/保留同一真值ImageNet类别。类别标签直接提供，未模拟CMI传输与额外资源成本。
- PSNR使用各约定定义的未钳制输出对原图计算；LPIPS采用clamp和normalize=True；CLIP采用clamp→PIL→官方预处理。token正确率始终对该流水线自己的发送ID计算，跨编码ID差异不作信道TER。
- 复用只限冻结来源清单中身份匹配的旧演示结果：计划、生成参数、PHY种子、权重、指标约定、源图、接收文件、恢复数组及原结果行均核对。冻结的复用来源清单允许旧研究以后继续，不改变本对照的执行身份。既有文件均未改写。

## 质量汇总状态

{quality}

汇总入口已通过四项独立检查：完整配对且输入顺序改变时正确汇总、缺少接收拒绝汇总、重复坐标不能掩盖缺项、缺少路径拒绝汇总。实际缓存另经输入哈希、最终ID、未掩码位置保持、有限指标和精确复用核验。

## 既有研究与范围

原多 SNR 研究仍为物理300/300、模型{old['completed_cases']}/300，本轮未改写其缓存或推进其组数；本对照拥有独立缓存和执行锁。100图×5次接收的冻结8 dB主实验仍为恢复与指标500/500。

主协议 SHA-256：`{session['primary_protocol_sha256']}`。
主汇总 SHA-256：`{session['primary_summary_sha256']}`。
独立对照计划 SHA-256：`{audit['plan_sha256']}`。

本研究沿用十张旧开发图，属于探索性配对对照；固定交织、oracle错误定位、理想正确token参考及完美类别条件不代表整篇论文的实际部署设置。此前无噪重建和固定掩码预测没有给出同一变化方向；{comparison_interpretation}原始图像、权重、逐接收数组均不进入Git白名单。

{provenance}
## 发布与续跑

{resume_text}发布提交、CI和精确Git树核验另写本地发布回执，避免自引用。草稿PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。
''', encoding='utf-8')
scripts = ['cache_stage57_comparison_phy.py','audit_stage57_comparison_phy.py',
           'run_stage57_convention_comparison.py','audit_stage57_convention_comparison.py',
           'comparison_aggregation.py','test_stage57_comparison_aggregation.py',
           'report_convention_comparison.py','plot_comparison_differences.py','sync_github_worktree_from_api.py']
if provenance:
    scripts.append('audit_checkpoint_metadata.py')
evidence = ['stage56_end_to_end_convention_plan.json','stage57_comparison_phy_lock.json',
            'stage57_comparison_phy_progress.json','stage57_comparison_phy_audit.json',
            'stage57_reuse_source_audit.json','stage57_comparison_execution_lock.json',
            'stage57_comparison_progress.json','stage57_comparison_audit.json',
            'stage57_aggregation_contract_verification.json',session_path.name,progress_name,audit_copy]
if (run/'stage58_plot_smoke_verification.json').exists():
    plot_test = read('stage58_plot_smoke_verification.json')
    assert plot_test['passed'] and plot_test['plot_script_sha256'] == sha(root/'plot_comparison_differences.py')
    evidence.append('stage58_plot_smoke_verification.json')
if provenance:
    evidence.append('stage58_checkpoint_metadata_audit.json')
    if (run/'stage58_upstream_weight_source_audit.json').exists():
        evidence.append('stage58_upstream_weight_source_audit.json')
entries = {f'code/{n}': (root/n).read_bytes() for n in scripts}
entries.update({f'evidence/{n}': (run/n).read_bytes() for n in evidence}); entries[f'reports/{report.name}'] = report.read_bytes()
entries.update({f'reports/{p.name}': p.read_bytes() for p in figure_paths})
manifest = {n: hashlib.sha256(data).hexdigest() for n,data in entries.items()}
entries['SHA256.json'] = json.dumps(manifest, indent=2).encode('utf-8')
archive = outputs/f'{prefix}_代码与证据.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for n,data in entries.items(): bundle.writestr(n,data)
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert all(hashlib.sha256(bundle.read(n)).hexdigest() == digest for n,digest in manifest.items())
print(json.dumps({'stage':stage,'counts':audit['counts'],'computed_cases':audit['computed_cases'],
    'reused_demo_cases':audit['reused_demo_cases'],'complete_pairs':audit['complete_pairs'],
    'archive_entries':len(entries),'archive_sha256':sha(archive),'report':str(report),'archive':str(archive),
    'figures':[str(p) for p in figure_paths]}, indent=2))
