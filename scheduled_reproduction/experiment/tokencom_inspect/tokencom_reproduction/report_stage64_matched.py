"""Publish bounded six-path coverage without incomplete-SNR quality means."""
import hashlib
import json
import zipfile
from datetime import datetime,timezone
from pathlib import Path
r=Path(__file__).resolve().parent;run=r/'runs';out=r.parents[2]/'outputs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((run/'stage62_matched_recovery_audit.json').read_text(encoding='utf-8'))
reference_spec=json.loads((run/'stage60_execution_spec.json').read_text(encoding='utf-8'))
assert sha(r/'evaluate_stage9_pretrained_maskgit.py')==reference_spec['class_mapping_script_sha256']
assert a['all_checks_passed'] and a['audit_script_sha256']==sha(r/'audit_stage62_matched_recovery.py')
assert sha(run/'stage47_complete_holdout_summary.json')=='c1590aef6aec5e4257a46ed79174dc8ce0f11b21d2bf7ee314a0804053d96979'
assert sha(run/'stage59_development_recovery_audit.json')=='b3810a4889c98108e18b88f421a8d817e1467a7d6c04bb096967683478e48436'
assert sha(run/'stage58_session_comparison_audit.json')=='3f3e50c51bb7d504bdc943d42d720185cff04b1f3121e196eec4f6b35096b273'
initial=json.loads((run/'stage64_window_start.json').read_text(encoding='utf-8'))
assert sha(run/'stage63_session_audit.json')==initial['initial_audit_sha256']
assert sha(r/'run_stage62_matched_recovery.py')==initial['runner_sha256']
assert sha(run/'stage62_matched_recovery_lock.json')==initial['lock_sha256']
assert all(sha(run/'stage62_matched_recovery'/name)==h for name,h in initial['initial_case_sha256'].items())
(run/'stage64_session_audit.json').write_bytes((run/'stage62_matched_recovery_audit.json').read_bytes())
n=a['completed_cases'];now=datetime.now(timezone.utc).isoformat()
dataset=json.loads((run/'stage64_author_dataset_protocol.json').read_text(encoding='utf-8'))
assert dataset['all_checks_passed'] and dataset['script_sha256']==sha(r/'audit_author_dataset_protocol.py')
prefix='TokCom_阶段64匹配填充帧续跑_2026-11-28'
coverage='\n'.join(f'| {s} | {n}/50 |' for s,n in a['counts_by_snr'].items())
from audit_matched_arq_comparison import summarize
comparison=summarize(r)
figures=[]
if comparison['complete']:
    from plot_matched_arq_comparison import plot
    figures=plot(comparison,out,prefix)
comparison_text=''
if comparison['complete_snr_groups']:
    table=['| SNR/dB | 路径 | 残余TER | PSNR/dB | LPIPS | CLIP | pixels/前向QAM符号 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for snr,group in comparison['complete_snr_groups'].items():
        for category in ('neural_means','arq_means'):
            for path,v in group[category].items():
                efficiency=('参考' if path=='ideal_correct_tokens' else f"{65536/1664:.6f}") if category=='neural_means' else f"{group['arq_forward_pixels_per_symbol'][path]:.6f}"
                table.append(f"| {snr} | {path} | {v['whole_image_ter']:.6%} | {v['psnr_raw']:.6f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} | {efficiency} |")
    comparison_text='## 完整分组的六路径与三种ARQ对照\n\n'+chr(10).join(table)+'\n\n质量先平均每图五次接收，再平均十图；成本按总pixels/总前向符号计算。所有路径对三种ARQ的逐图配对差值均保存在机器回执中，不选赢家、不作显著性结论。理想正确token仅作质量参考，不赋予可实现传输成本。未计反馈、包头、类别信息空口开销。\n'
    directions=['| 生成路径 | PSNR高于直接解码 | LPIPS低于直接解码 | CLIP高于直接解码 | TER低于直接解码 |',
                '|---|---:|---:|---:|---:|']
    for path in ('oracle_without_cmi','oracle_with_cmi','receiver_crc_without_cmi','receiver_crc_with_cmi'):
        counts=[sum((g['neural_means'][path][k]>g['neural_means']['direct'][k]) if higher else (g['neural_means'][path][k]<g['neural_means']['direct'][k]) for g in comparison['complete_snr_groups'].values()) for k,higher in (('psnr_raw',True),('lpips_normalized',False),('clip_clipped',True),('whole_image_ter',False))]
        directions.append('| '+path+' | '+' | '.join(f'{n}/6' for n in counts)+' |')
    comparison_text+='\n## 描述性方向核对\n\n'+chr(10).join(directions)+'\n\n这些是六个固定SNR的方向统计，不是独立重复或显著性检验。oracle路径依赖真实错误位置，CRC路径保留实际CRC检测结果；本轮结果不支持把oracle掩码收益直接外推到CRC接收实现。token精确匹配与像素/语义质量需分别解读。\n'

report=out/(prefix+'.md')
report.write_text(f'''# 阶段64：相同填充帧格式的六路径模型恢复

记录时间：{now}。本轮新增 {n-initial["initial_cases"]} 组，累计审计 {n} 组接收，每组保存六路径最终token和质量记录，共 {6*n} 行；计划总量300组。未新增物理模拟，使用上一轮已冻结的300份首发输入。

| SNR/dB | 完成接收 |
|---|---:|
{coverage}

{comparison_text}

{('![完整端点对照]('+prefix+'_完整对照.svg)' if figures else '')}

六路径：直接解码、oracle掩码有无类别信息、接收端CRC掩码有无类别信息、理想正确token。四种生成路径使用各自正确掩码，同一掩码两种类别模式共享输入；生成种子20261124，32步。保留原demo输入x与原始解码y、权重和指标约定。CRC错误接受造成的未掩码错误保留，不能利用发送端真值偷偷修正CRC掩码。

每帧首发1664个前向QAM符号，与stage61实际包ARQ相同；未计反馈、包头和类别信息传输成本。oracle仍需要发送端真值。本轮不复用旧1632符号研究的生成端点。每SNR六路径均齐备50组后才能汇总质量；当前完整SNR组为{len(a['complete_snr_groups'])}。未完整分组不发布质量均值。

既有六项纯逻辑检查通过，包括CRC未检错误保持、oracle未掩码位置保持、错误计数、非有限指标和缺路径拒绝。执行锁绑定入口、审计器、检查函数、测试、计划、输入审计和指标身份。审计复核最终token、两种掩码、源文件SHA256、有限指标和理想干净参考。

原主实验500组、固定分组300组、随机分组300组、输入约定对照100组保持独立。整篇论文复现未完成，仍是旧Imagenette十图开发实验。

## 下一阶段数据协议证据

本轮只读核对作者固定提交{dataset['official_commit']}的演示notebook：默认class_indices为[503,498]、每类40张、SNR为[5,7]；数据加载单元另使用固定目录，文件列表未排序后random.sample。该证据仅说明此版本演示配置，不能据此确定论文完整100类清单、测试划分或精确图像身份。未修改作者源码、未下载新数据，也未改变本轮十图设置。下一步优先补齐100类synset清单、图像ID及SHA256、采样顺序与种子、Cheng配置和完整资源开销协议。

{('六路径恢复已齐备300组，保存完整对照并推进数据集及剩余资源开销对齐。' if a['complete'] else '下一轮继续同一执行锁下的缺失组，随后重新审计。')}保持执行锁及已有缓存；依然只汇总完整SNR，最终与stage61同帧格式的三种ARQ端点比较。GitHub草稿PR：https://github.com/zhoujinyi8/tokencomm/pull/2 。
''',encoding='utf-8')
sources=['run_stage62_matched_recovery.py','audit_stage62_matched_recovery.py','matched_recovery_checks.py','test_matched_recovery_checks.py','report_stage64_matched.py','audit_matched_arq_comparison.py','plot_matched_arq_comparison.py','audit_author_dataset_protocol.py']
evidence=['stage64_author_dataset_protocol.json','stage64_matched_arq_comparison.json','stage63_session_audit.json','stage61_arq_metric_audit.json','stage61_packet_arq_audit.json','stage64_window_start.json','stage64_session_audit.json','stage62_session_audit.json','stage62_session_progress.json','stage61_matched_padded_recovery_plan.json','stage61_matched_padded_input_lock.json','stage61_matched_padded_input_audit.json','stage62_matched_recovery_lock.json','stage62_matched_recovery_progress.json','stage62_matched_recovery_audit.json']
entries={f'code/{n}':(r/n).read_bytes() for n in sources}
entries.update({f'evidence/{n}':(run/n).read_bytes() for n in evidence});entries['reports/'+report.name]=report.read_bytes()
entries.update({'reports/'+p.name:p.read_bytes() for p in figures})
manifest={n:hashlib.sha256(v).hexdigest() for n,v in entries.items()};entries['SHA256.json']=json.dumps(manifest,indent=2).encode()
archive=out/(prefix+'_代码与证据.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for n,v in entries.items():z.writestr(n,v)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
print(json.dumps(dict(completed_cases=a['completed_cases'],rows=6*a['completed_cases'],report=str(report),archive=str(archive),archive_sha256=sha(archive))))
