"""Record this hourly continuation and CI verification without interim model means."""
import json
import hashlib
import zipfile
from datetime import datetime,timezone
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'runs';outputs=root.parents[2]/'outputs'
recovery=json.loads((run/'stage46_recovery_progress.json').read_text(encoding='utf-8'))
ci=json.loads((run/'stage48_ci_verification.json').read_text(encoding='utf-8'))
metrics=json.loads((run/'stage47_metric_file_audit.json').read_text(encoding='utf-8')) if (run/'stage47_metric_file_audit.json').exists() else {}
current=recovery['completed_recovery_cases'];metric_count=metrics.get('metric_receptions_saved',0)
result={'session_start_utc':'2026-11-21T22:43:56.511Z','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'initial_recovery_cases':210,'recovery_cases':current,'new_recovery_cases':current-210,
        'complete_five_trial_images':recovery['complete_five_trial_images'],
        'metric_cases':metric_count,'expected_cases':500,'paths_per_metric_case':6,
        'model_performance_aggregated':False,'frozen_scientific_settings_changed':False,
        'ci_verification':ci}
(run/'stage48_session_progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
report=outputs/'TokCom_第四十八阶段留出续跑与GitHub检查_2026-11-22.md'
report.write_text(f'''# TokCom 第四十八阶段：留出续跑与 GitHub 检查

从上一轮的 210 组恢复继续执行第四十四阶段冻结协议，本轮新增 {current-210} 组，当前累计 {current}/500 组；五轮齐全图像 {recovery['complete_five_trial_images']}/100 张。100 张作者 token 编码与 500 组物理接收缓存已经完成。没有据新留出结果修改模型、种子、掩码、门控比例或指标口径。

逐组图像指标缓存已保存 {metric_count}/500 组，每组包含直接解码、真值掩码恢复、CRC 恢复和三档软门控，共六条路径。指标入口在首次执行前锁定代码与权重身份，保存完整接收组后才写入正式文件。当前没有汇总部分样本的模型性能；全量汇总要求 100 图、五轮均齐全。

## 本轮仓库检查修复

1. 修复仓库原有 LPIPS 训练辅助模块的未定义打印引用：调用已导入的 lpips.print_network。用替身网络验证打印分支，不下载或加载权重。此修改没有改变正在运行的冻结恢复源码或所用指标模型。
2. 增加独立 requirements-ci.txt，声明已有测试导入的 torchvision、matplotlib、transformers、peft 及检查工具。生产缓存脚本不再因 *_test.py 后缀被误收集。
3. 精确标记依赖数据、检查点或历史冻结资产的 11 项集成测试；保留其断言，可在恢复资产后运行 pytest -m integration。独立的 V4 梯度与状态往返测试从检查点初始化中移出，继续在干净环境执行。

GitHub 检查通过：严格 flake8 通过，单元测试 96 项通过、2 项跳过，11 项资产集成测试未运行。不能把单元检查通过解释为历史数据、检查点或冻结哈希审计均通过；历史 Stage19 的源码身份差异仍需在恢复该实验时处理，没有改写其预期哈希。检查记录：https://github.com/zhoujinyi8/tokencomm/actions/runs/44959937005 。

本轮 GitHub 仍更新同一草稿 PR：https://github.com/zhoujinyi8/tokencomm/pull/1 。更新源码、协议、进度和审计，原图、大权重和生成缓存保留本地。实际最新提交链接由本轮回复及同步回执记录。

## 后续

继续同一执行锁下的缺失恢复组和指标缓存，完成全体后再按图像聚合五轮信道。新留出结果已进入固定评估，不能用来继续选择门控阈值。当前仍是单一 8 dB、固定生成种子及真实类别假设，尚未完成论文 TCE/反馈协议或多 SNR 复现。
''',encoding='utf-8')
names=['report_stage48_session.py','audit_stage47_metric_files.py','evaluate_stage47_frozen_metrics.py',
       'runs/stage48_session_progress.json','runs/stage48_ci_verification.json',
       'runs/stage46_recovery_progress.json','runs/stage46_output_audit.json']
for name in ['stage47_metrics_lock.json','stage47_metric_file_audit.json']:
    if (run/name).exists():names.append('runs/'+name)
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
archive_path=outputs/'TokCom_stage48_holdout_continuation_ci.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,base in items:archive.write(base/name,name)
    archive.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest()==value for name,value in manifest.items())
print(json.dumps({k:v for k,v in result.items() if k!='ci_verification'},indent=2))
