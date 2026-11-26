"""Render a bounded random-packet session from immutable plans and final audits."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session',required=True,type=Path)
args=parser.parse_args();root=Path(__file__).resolve().parent;run=root/'runs';outputs=root.parents[2]/'outputs'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
session_path=(root/args.session).resolve();assert session_path.parent==run.resolve()
session=json.loads(session_path.read_text(encoding='utf-8'));stage=session['stage'];assert stage>=60
spec=read('stage60_execution_spec.json');plan=read('stage59_randomized_packetization_plan.json')
phy=read('stage60_randomized_phy_audit.json');model=read('stage60_randomized_recovery_audit.json')
paired=read('stage60_paired_packetization_audit.json');fixed_phy=read('stage59_development_phy_audit.json')
fixed_model=read('stage59_development_recovery_audit.json');runtime=read(f'stage{stage}_runtime_identity.json')
assert phy['all_checks_passed'] and phy['complete'] and model['all_checks_passed'] and paired['all_checks_passed']
assert fixed_phy['complete'] and fixed_model['complete'] and fixed_model['completed_cases']==300
assert phy['plan_sha256']==model['plan_sha256']==paired['plan_sha256']==sha(run/'stage60_execution_spec.json')
assert sha(run/'stage59_randomized_packetization_plan.json')==spec['preregistered_plan_sha256']==session['preregistered_plan_sha256']
assert sha(run/'stage59_development_recovery_audit.json')==spec['baseline_complete_audit_sha256']==session['baseline_complete_audit_sha256']
assert sha(run/'stage59_development_phy_audit.json')==spec['baseline_phy_audit_sha256']
assert paired['randomized_audit_sha256']==sha(run/'stage60_randomized_recovery_audit.json')
assert phy['audit_script_sha256']==sha(root/'audit_stage60_randomized_phy.py')
assert model['audit_script_sha256']==sha(root/'audit_stage60_randomized_recovery.py')
assert paired['audit_script_sha256']==sha(root/'audit_stage60_paired_packetization.py')
assert sha(run/'stage44_frozen_protocol.json')==session['primary_protocol_sha256']
assert sha(run/'stage47_complete_holdout_summary.json')==session['primary_summary_sha256']
assert sha(run/'stage58_session_comparison_audit.json')==session['comparison_frozen_audit_sha256']
assert runtime['all_checks_passed'] and sha(run/f'stage{stage}_runtime_identity.json')==session['runtime_identity_sha256']
assert runtime['execution_spec_sha256']==sha(run/'stage60_execution_spec.json')
assert runtime['runner_sha256']==sha(root/'run_stage60_randomized_recovery.py')
assert runtime['class_mapping_script_sha256']==spec['class_mapping_script_sha256']
phy_runtime=read('stage60_phy_runtime_identity.json')
assert phy_runtime['all_checks_passed'] and phy_runtime['simulator_sha256']==spec['phy_simulator_sha256']
assert phy_runtime['packages']=={'tensorflow':'2.15.1','sionna':'0.19.2','numpy':'1.26.4'}
for namespace,audit,suffix in [('stage54_development_recovery',fixed_model,'.json'),
                              ('stage54_development_phy',fixed_phy,'.npz'),
                              ('stage60_randomized_phy',phy,'.npz'),
                              ('stage60_randomized_recovery',model,'.json')]:
    for name,digest in audit['case_sha256'].items():assert sha(run/namespace/name)==digest
contracts=[('stage60_packetization_contract.json',22,'helper_sha256','randomized_packetization.py','test_randomized_packetization.py'),
           ('stage60_model_audit_contract.json',4,'audit_script_sha256','audit_stage60_randomized_recovery.py','test_stage60_model_audit.py'),
           ('stage60_paired_aggregation_contract.json',5,'helper_sha256','paired_packetization_aggregation.py','test_paired_packetization_aggregation.py')]
for name,count,key,source,test_source in contracts:
    contract=read(name);assert contract['passed']==count and contract['failed']==0
    assert contract[key]==sha(root/source)
    assert contract['test_script_sha256']==sha(root/test_source)
smoke=read('stage60_phy_roundtrip_smoke.json');noiseless=read('stage60_noiseless_phy_verification.json')
assert smoke['passed']==noiseless['passed']==2 and not noiseless['added_channel_noise']
assert smoke['script_sha256']==sha(root/'cache_stage60_randomized_phy.py')
assert noiseless['script_sha256']==sha(root/'test_stage60_noiseless_phy.py')
assert noiseless['execution_spec_sha256']==sha(run/'stage60_execution_spec.json')
diagnostics=smoke['diagnostic_phy_simulations']+noiseless['diagnostic_noiseless_transmissions']
now=datetime.now(timezone.utc);date=now.astimezone(timezone(timedelta(hours=8))).date().isoformat()
prefix=f'TokCom_阶段{stage}随机交织对照_{date}'
result=dict(session,recorded_utc=now.isoformat(),randomized_phy_cases=phy['completed_cases'],
            randomized_model_cases=model['completed_cases'],randomized_model_counts_by_snr=model['counts_by_snr'],
            new_randomized_phy_cases=phy['completed_cases']-session['initial_randomized_phy_cases'],
            new_randomized_model_cases=model['completed_cases']-session['initial_randomized_model_cases'],
            diagnostic_phy_transmissions_total=diagnostics,
            new_diagnostic_phy_transmissions=diagnostics-session.get('initial_diagnostic_phy_transmissions',diagnostics),
            remaining_randomized_model_cases=300-model['completed_cases'],complete_randomized_study=model['complete'],
            complete_paired_quality_snr_groups=paired['paired_quality_snr_groups'],
            baseline_model_cases_unchanged=300,primary_recovery_cases_unchanged=500,
            comparison_cases_unchanged=100,primary_holdout_changed=False,selected_pipeline=None,
            full_paper_reproduction_complete=False)
audit_copies=[]
for label,source in [('phy','stage60_randomized_phy_audit.json'),('model','stage60_randomized_recovery_audit.json'),
                     ('paired','stage60_paired_packetization_audit.json')]:
    name=f'stage{stage}_session_{label}_audit.json';(run/name).write_bytes((run/source).read_bytes());audit_copies.append(name)
coverage=['| SNR/dB | 固定基线模型/50 | 随机PHY/50 | 随机四路径模型/50 | 配对质量可汇总 |',
          '|---|---:|---:|---:|---|']
physical=['| SNR/dB | 固定包PER | 随机包PER | 固定接收TER | 随机接收TER | 随机CRC失败率 |',
          '|---|---:|---:|---:|---:|---:|']
for snr in spec['snr_db']:
    key=str(snr);f=fixed_phy['complete_snr_groups'][key];p=phy['complete_snr_groups'][key]
    coverage.append(f"| {snr:g} | 50 | {phy['counts_by_snr'][key]} | {model['counts_by_snr'][key]} | {'是' if key in paired['groups'] else '否'} |")
    physical.append(f"| {snr:g} | {f['true_packet_error_rate']:.6%} | {p['true_packet_error_rate']:.6%} | {f['received_token_error_rate']:.6%} | {p['received_token_error_rate']:.6%} | {p['crc_failure_rate']:.6%} |")
quality='随机模型尚无完整50接收的SNR组，不输出部分组的模型质量均值或固定/随机质量差值。'
if paired['groups']:
    rows=['| SNR/dB | 分支 | 路径 | 正确token/256 | 残余TER | PSNR/dB | LPIPS | CLIP |',
          '|---|---|---|---:|---:|---:|---:|---:|']
    for key,group in paired['groups'].items():
        assert group['receptions_per_stream']==50 and group['images']==10
        for branch in ('fixed','randomized'):
            for path in plan['paths']:
                v=group['means_over_images'][branch][path]
                rows.append(f"| {key} | {branch} | {path} | {v['correct_tokens']:.6f} | {v['whole_image_ter']:.6%} | {v['psnr_raw']:.6f} | {v['lpips_normalized']:.6f} | {v['clip_clipped']:.6f} |")
    quality=chr(10).join(rows)+'\n\n先对每图五次接收平均，再平均十图；全部四路径与逐图配对差值保存在配对审计JSON。未完整的SNR不参与质量汇总，不选择最佳流水线。'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
for branch,audit,marker in [('Fixed',fixed_phy,'o'),('Randomized',phy,'s')]:
    for axis,key,title in [(axes[0],'true_packet_error_rate','True payload packet error rate'),
                           (axes[1],'received_token_error_rate','Received token error rate')]:
        axis.plot(spec['snr_db'],[100*audit['complete_snr_groups'][str(s)][key] for s in spec['snr_db']],marker=marker,label=branch)
        axis.set(xlabel='SNR (dB)',ylabel='Error rate (%)',title=title,xticks=spec['snr_db']);axis.grid(alpha=.25);axis.legend()
fig.suptitle('Complete PHY streams: 10 old images x 5 receptions per SNR\nFixed versus preregistered random packet assignment; model recovery reported separately',fontsize=11)
fig.supxlabel('All PHY groups contain 50 receptions. This figure does not establish generative recovery quality.',fontsize=9)
figures=[]
for suffix in ('svg','png'):
    path=outputs/f'{prefix}_PHY.{suffix}';fig.savefig(path,dpi=160);figures.append(path)
plt.close(fig)
result['phy_figure_sha256']={p.name:sha(p) for p in figures}
progress_name=f'stage{stage}_session_progress.json';(run/progress_name).write_text(json.dumps(result,indent=2),encoding='utf-8')
report=outputs/f'{prefix}.md'
report.write_text(f'''# TokCom 阶段{stage}：独立随机交织对照

记录时间：{now.isoformat()}（UTC；Asia/Shanghai日期用于文件名）。执行预先保存的stage59_randomized_packetization_plan.json，既有冻结结果保持独立。

## 实际新增与覆盖

本轮新增正式随机PHY {result['new_randomized_phy_cases']} 组、四路径模型恢复 {result['new_randomized_model_cases']} 组。当前随机PHY {phy['completed_cases']}/300、模型 {model['completed_cases']}/300，剩余模型 {result['remaining_randomized_model_cases']} 组。另有本轮新增诊断传输 {result['new_diagnostic_phy_transmissions']} 次，诊断累计 {diagnostics} 次：两次80dB高SNR往返，以及两次没有添加信道噪声的CRC/卷积码/QAM往返；诊断不计入正式300组。新随机模型组均实际计算，没有转用固定基线或约定对照的恢复结果。

{chr(10).join(coverage)}

## 独立输入与坐标规则

同十张旧开发图、每图五次接收、6/6.5/7/7.5/8/9dB。每图固定PCG64种子202611260+图序号的256位置置换，五次接收和六SNR共享该图置换。发送前置换，接收ID、oracle与CRC token掩码经逆置换恢复到原图坐标后再解码或生成。错误包按置换后的实际四token成员判断，不能按恢复后的相邻位置重建包。

保留原VQGAN/MaskGIT权重、RGB/Resize((256,256))/ToTensor、演示编码输入x与原解码输出y、原噪声种子公式202611240+100×图序号+接收序号、生成种子20261124和32步参数。w=0/1分别去除/保留同一真值类别；同一随机接收的两种类别模式共享自身oracle掩码。固定和随机分支发送符号及错误包成员可以不同，不强行共用错误位置。PSNR用原始输出，LPIPS用clamp与normalize=True，CLIP用clamp→PIL→官方预处理。

CRC和卷积编码仍按包处理，16-QAM仍采用原全帧映射：payload2560bit、CRC704bit、编码6528bit、1632符号。此处只改包位置分配，没有引入独立包补位、实际重传或CMI传输成本。

## 完整物理接收

{chr(10).join(physical)}

![完整物理接收统计]({prefix}_PHY.svg)

每个SNR、每分支均为50接收，包统计分母3200、token统计分母12800。物理误差分开审计，不从这些统计推断MaskGIT恢复收益；未进行置换种子选择或显著性检验。

## 模型质量与配对门槛

{quality}

## 验证与原实验

22项置换/掩码检查、4项实际模型审计隔离检查、5项配对汇总门槛检查均通过；两次高SNR与两次精确无噪链路检查通过。精确无噪检查不添加信道噪声，demapper采用正方差0.01得到有限软LLR，该值不是新增AWGN。模型审计重新核对最终ID、未掩码位置保持、有限指标及理想正确token参考与固定基线一致性；作者TER_af不作为残余TER。

固定多SNR基线300/300保持不变，冻结8dB主实验恢复与指标500/500保持不变，输入约定对照100/100保持不变。主协议SHA256：`{session['primary_protocol_sha256']}`；主汇总：`{session['primary_summary_sha256']}`；预先计划：`{spec['preregistered_plan_sha256']}`；首次新推理前已绑定完整固定基线审计：`{spec['baseline_complete_audit_sha256']}`。代码、输入与执行锁均在独立命名空间，历史报告不改写。

本研究为旧Imagenette开发图的随机包分配敏感性分析，不是ImageNet100数据集复现，也未恢复论文未公开的具体随机种子。类别条件完美提供、oracle需要发送端真值、理想正确token不是重传结果；资源等价基线与整篇论文复现仍未完成。

## 发布与续跑

保持执行规范、PHY/模型执行锁及原缓存，运行run_stage60_randomized_recovery.py继续缺失组，再运行audit_stage60_randomized_recovery.py和audit_stage60_paired_packetization.py。每SNR两分支均齐备50接收后才汇总全部四路径与逐图配对差值。发布提交、实际CI和精确Git对象核验另写本地回执，避免自引用。草稿PR：https://github.com/zhoujinyi8/tokencomm/pull/{session['github_pull_request']} 。

PHY缓存和精确无噪诊断使用独立TensorFlow2.15.1/Sionna0.19.2环境；模型恢复及结果审计使用主PyTorch环境。既有正式PHY已完整时不重复模拟，模型入口自动跳过身份匹配的完整组。
''',encoding='utf-8')
scripts=['randomized_packetization.py','test_randomized_packetization.py','cache_stage60_randomized_phy.py',
         'audit_stage60_randomized_phy.py','run_stage60_randomized_recovery.py','audit_stage60_randomized_recovery.py',
         'test_stage60_model_audit.py','test_stage60_noiseless_phy.py','paired_packetization_aggregation.py',
         'test_paired_packetization_aggregation.py','audit_stage60_paired_packetization.py','report_randomized_session.py',
         'smoke_stage25_sionna_link.py','paper_tce_reference.py','evaluate_stage9_pretrained_maskgit.py','sync_github_worktree_from_api.py']
evidence=['stage59_randomized_packetization_plan.json','stage59_development_recovery_audit.json','stage59_development_phy_audit.json',
          'stage44_frozen_protocol.json','stage47_complete_holdout_summary.json','stage58_session_comparison_audit.json',
          'stage46_environment.json','stage47_metrics_lock.json','stage54_development_multisnr_plan.json',
          'stage60_execution_spec.json','stage60_phy_runtime_identity.json','stage60_randomized_phy_lock.json','stage60_randomized_phy_progress.json',
          'stage60_randomized_phy_audit.json','stage60_phy_roundtrip_smoke.json','stage60_noiseless_phy_verification.json',
          'stage60_packetization_contract.json','stage60_randomized_recovery_lock.json','stage60_randomized_recovery_progress.json',
          'stage60_randomized_recovery_audit.json','stage60_model_audit_contract.json','stage60_paired_aggregation_contract.json',
          'stage60_paired_packetization_audit.json',session_path.name,f'stage{stage}_runtime_identity.json',progress_name]+audit_copies
entries={f'code/{n}':(root/n).read_bytes() for n in scripts}
entries.update({f'evidence/{n}':(run/n).read_bytes() for n in evidence})
entries[f'reports/{report.name}']=report.read_bytes();entries.update({f'reports/{p.name}':p.read_bytes() for p in figures})
manifest={n:hashlib.sha256(v).hexdigest() for n,v in entries.items()}
entries['SHA256.json']=json.dumps(manifest,indent=2).encode('utf-8')
archive=outputs/f'{prefix}_代码与证据.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for n,v in entries.items():bundle.writestr(n,v)
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None and all(hashlib.sha256(bundle.read(n)).hexdigest()==v for n,v in manifest.items())
print(json.dumps(dict(stage=stage,phy_cases=phy['completed_cases'],model_cases=model['completed_cases'],
    new_phy=result['new_randomized_phy_cases'],new_model=result['new_randomized_model_cases'],diagnostic_transmissions=diagnostics,
    paired_quality_snr_groups=paired['paired_quality_snr_groups'],archive_entries=len(entries),archive_sha256=sha(archive),
    report=str(report),archive=str(archive),figures=[str(p) for p in figures]),indent=2))
