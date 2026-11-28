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
initial=json.loads((run/'stage63_window_start.json').read_text(encoding='utf-8'))
assert sha(run/'stage62_session_audit.json')==initial['initial_audit_sha256']
assert sha(r/'run_stage62_matched_recovery.py')==initial['runner_sha256']
assert sha(run/'stage62_matched_recovery_lock.json')==initial['lock_sha256']
assert all(sha(run/'stage62_matched_recovery'/name)==h for name,h in initial['initial_case_sha256'].items())
(run/'stage63_session_audit.json').write_bytes((run/'stage62_matched_recovery_audit.json').read_bytes())
n=a['completed_cases'];now=datetime.now(timezone.utc).isoformat()
prefix='TokCom_阶段63匹配填充帧续跑_2026-11-28'
coverage='\n'.join(f'| {s} | {n}/50 |' for s,n in a['counts_by_snr'].items())
report=out/(prefix+'.md')
report.write_text(f'''# 阶段63：相同填充帧格式的六路径模型恢复

记录时间：{now}。本轮新增 {n-initial["initial_cases"]} 组，累计审计 {n} 组接收，每组保存六路径最终token和质量记录，共 {6*n} 行；计划总量300组。未新增物理模拟，使用上一轮已冻结的300份首发输入。

| SNR/dB | 完成接收 |
|---|---:|
{coverage}

六路径：直接解码、oracle掩码有无类别信息、接收端CRC掩码有无类别信息、理想正确token。四种生成路径使用各自正确掩码，同一掩码两种类别模式共享输入；生成种子20261124，32步。保留原demo输入x与原始解码y、权重和指标约定。CRC错误接受造成的未掩码错误保留，不能利用发送端真值偷偷修正CRC掩码。

每帧首发1664个前向QAM符号，与stage61实际包ARQ相同；未计反馈、包头和类别信息传输成本。oracle仍需要发送端真值。本轮不复用旧1632符号研究的生成端点。每SNR六路径均齐备50组后才能汇总质量；当前完整SNR组为{len(a['complete_snr_groups'])}。未完整分组不发布质量均值。

既有六项纯逻辑检查通过，包括CRC未检错误保持、oracle未掩码位置保持、错误计数、非有限指标和缺路径拒绝。执行锁绑定入口、审计器、检查函数、测试、计划、输入审计和指标身份。审计复核最终token、两种掩码、源文件SHA256、有限指标和理想干净参考。

原主实验500组、固定分组300组、随机分组300组、输入约定对照100组保持独立。整篇论文复现未完成，仍是旧Imagenette十图开发实验。

下一轮运行run_stage62_matched_recovery.py继续缺失组，再运行audit_stage62_matched_recovery.py。保持执行锁及已有缓存；依然只汇总完整SNR，最终与stage61同帧格式的三种ARQ端点比较。GitHub草稿PR：https://github.com/zhoujinyi8/tokencomm/pull/2 。
''',encoding='utf-8')
sources=['run_stage62_matched_recovery.py','audit_stage62_matched_recovery.py','matched_recovery_checks.py','test_matched_recovery_checks.py','report_stage63_matched.py']
evidence=['stage63_window_start.json','stage63_session_audit.json','stage62_session_audit.json','stage62_session_progress.json','stage61_matched_padded_recovery_plan.json','stage61_matched_padded_input_lock.json','stage61_matched_padded_input_audit.json','stage62_matched_recovery_lock.json','stage62_matched_recovery_progress.json','stage62_matched_recovery_audit.json']
entries={f'code/{n}':(r/n).read_bytes() for n in sources}
entries.update({f'evidence/{n}':(run/n).read_bytes() for n in evidence});entries['reports/'+report.name]=report.read_bytes()
manifest={n:hashlib.sha256(v).hexdigest() for n,v in entries.items()};entries['SHA256.json']=json.dumps(manifest,indent=2).encode()
archive=out/(prefix+'_代码与证据.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for n,v in entries.items():z.writestr(n,v)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
print(json.dumps(dict(completed_cases=a['completed_cases'],rows=6*a['completed_cases'],report=str(report),archive=str(archive),archive_sha256=sha(archive))))
