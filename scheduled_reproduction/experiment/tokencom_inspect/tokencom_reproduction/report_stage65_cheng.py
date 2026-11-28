"""Publish audited codec engineering and explicitly scoped proxy training evidence."""
import hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
r=Path(__file__).resolve().parent;run=r/'runs';out=r.parents[2]/'outputs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
a=read('stage65_cheng_audit.json');budget=read('stage65_cheng_packet_budget.json');contracts=read('stage65_contracts.json')
gaps=read('stage65_protocol_gaps.json');gradient=read('stage65_gradient_smoke_audit.json');pilot=read('stage65_rate_pilot_audit.json')
assert a['all_checks_passed'] and a['complete'] and a['completed_cases']==20
assert a['audit_script_sha256']==sha(r/'audit_stage65_cheng_diagnostic.py')
assert budget['all_checks_passed'] and budget['source_audit_sha256']==sha(run/'stage65_cheng_audit.json') and budget['cases']==20
assert gradient['all_checks_passed'] and gradient['optimizer_updates']==1
assert pilot['all_checks_passed'] and pilot['source_result_sha256']==sha(run/'stage65_rate_pilot.json')
assert contracts['new_checks_passed']==23 and contracts['failed']==0
for n,h in contracts['source_sha256'].items():assert sha(r/n)==h
for n,h in a['case_sha256'].items():assert sha(run/'stage65_cheng_cases'/n)==h
for n,h in gaps['legacy_artifact_sha256_unchanged'].items():assert sha(run/n)==h
prefix='TokCom_阶段65Cheng码流与低码率准备_2026-11-28'
from plot_stage65_codec_rates import plot
figures=plot(r,out,prefix,a)
table=['| 模型（stock MSE q1） | 实际熵流bpp | 含自定义头bpp | 似然估计bpp | 相对0.04目标 | PSNR/dB | LPIPS | CLIP | 平均首发QAM符号 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for model,v in a['complete_model_groups'].items():
    m=v['means'];q=budget['complete_model_groups'][model]['forward_qam_symbols']
    table.append(f"| {model} | {m['entropy_payload_bpp']:.6f} | {m['total_stored_bpp']:.6f} | {m['estimated_likelihood_bpp']:.6f} | {v['payload_rate_ratio_to_paper_target']:.3f}倍 | {m['psnr_decoded']:.3f} | {m['lpips_normalized']:.6f} | {m['clip_clipped']:.6f} | {q:.1f} |")
pilot_text=f"固定预算代理试点已更新 {pilot['updates_completed']}/100 次；校准 {pilot['calibration_cases']}/10 张，独立解码复核 {len(pilot['fresh_decoder_checks'])}/10 张。"
if pilot['mean_entropy_payload_bpp'] is not None:
    pilot_text+=f"校准集实际熵流均值 {pilot['mean_entropy_payload_bpp']:.6f} bpp；是否落入预先指定0.04±10%的诊断窗口：{pilot['within_diagnostic_rate_tolerance']}。该容差不是作者给出的判据。"
    pilot_cases=read('stage65_rate_pilot.json')['calibration_cases']
    pilot_psnr=sum(v['psnr_decoded'] for v in pilot_cases)/len(pilot_cases)
    pilot_text+=f"同十图stock Anchor PSNR为{a['complete_model_groups']['cheng2020-anchor']['means']['psnr_decoded']:.3f}dB，代理试点为{pilot_psnr:.3f}dB，压缩率变化伴随失真变化。本试点仅做码率/PSNR校准，未作语义或感知性能结论。"
report=out/(prefix+'.md')
report.write_text(f'''# 阶段65：Cheng实际码流、低码率准备和数据协议

记录时间：{datetime.now(timezone.utc).isoformat()}。

## 已完成的stock码流诊断

独立Cheng环境完成两种公开Cheng变体、同一批十张旧开发图、各一个最低质量档位，共20/20个干净码流案例。原生rANS及CDF后端8项检查通过；两次独立进程仅用完整熵流、形状和固定权重解码，未读取原图、未调用编码器，像素最大差均为0。实际码率由两个完整熵流的字节数计算，似然估计另列。没有训练这些stock模型，也没有给本项诊断加入有噪信道或重传。

{chr(10).join(table)}

![实际码率与似然估计]({prefix}_实际码率.svg)

[TokCom arXiv v5第VI节](https://arxiv.org/html/2502.12096v5#S6)说明Cheng基线经过微调，目标约为0.04 bpp。此处公开q1模型仍约为目标的3.2倍，不能把它们的重建质量直接当作论文的等码率Cheng基线，也未将它们与TokCom作同码率优劣结论。作者的精确Cheng变体、微调权重、训练数据与训练配方仍待确定。

## 容器与首发账本

CH65是本工程定义的34字节头，包含模型/质量标识、图像与熵流形状、两个流长度；它不是已确定的作者报头。上表含头码率与纯熵流分开记录。账本把完整容器按40bit负载分包，尾部补零，逐包CRC11、半率卷积编码、再补2bit形成104bit/26个QAM符号。20个案例均经字节往返和整数账本核对；这里只计算首发资源，没有模拟信道、反馈或CMI。

## 单步训练通路与独立代理校准

单步训练smoke从原始Anchor q1开始，用单独训练图完成一次Adam主参数/辅助量化参数更新；122组主参数发生变化，独立审计发现104个同名浮点张量与stock检查点不同，重载后的解码像素一致。没有覆盖stock权重。该检查不表示已达到目标码率。

随后另立固定预算代理计划：从已有训练集选每类10张、共100张，按SHA排序和固定类间交替顺序，100次更新；lambda=0.0001、主学习率0.0001、辅助学习率0.001、batch=1、固定随机种子。超参数是工程选择，不是已确认的作者配方。按580条现有验证/测试记录的字节、像素和源文件身份排除训练重叠；优化不使用旧开发评估图或冻结主实验图。只在预算结束后对十张旧开发图做码率校准，不选最佳检查点。检查点保存模型、两个优化器和Torch RNG状态。

{pilot_text}

这仍是十类代理数据上的开发校准，不是百类论文复现或独立最终测试；即使进入诊断码率窗口，论文基线资格仍为False。

## 数据与版本证据

作者当前公开HEAD仍是0422ed32dcde3347e0bb6697f343f7a9ff7b3214；演示notebook的Git对象与首个公开版本相同，默认仅[503,498]两类、每类40图，目录文件未排序后随机取样。现有十类开发集和两类演示均被百类结构门槛明确拒绝。新增清单工具检查100个唯一synset、逐文件SHA256、跨split同图和冻结评估图进入训练等问题；结构合格仍不等于作者数据身份已确认。实际论文百类图像本轮核验数量为0。

[作者公开问题#1](https://github.com/liqiao19/TokenCom_Code/issues/1)提及TER复现疑问，本轮只读检查未见评论回复；没有向作者发送消息。arXiv v5、期刊版和代码版本的完全对应关系仍需核实。下一阶段还需要百类清单、分割/图像身份、准确微调配置和完整反馈/报头/CMI资源协议。

## 环境与校验

CompressAI 1.2.8源码和LLVM-MinGW归档按完整SHA256核验，两个上游C++扩展在工作区编译，源算法未修改。依赖装入独立venv，主TokCom环境保持原状。首次Attention响应内容摘要不符，已隔离且未用于推理；独立复取并固定正确完整摘要后才冻结计划。原主协议、主汇总、固定多SNR、输入约定和完整匹配帧结果均与上次发布字节摘要一致。

新增纯逻辑检查23项：容器8、分包6、数据清单9，均通过；与原20项合计43项加入CI。原生后端8项和实际模型/码流检查需要本地资产，另行记录。GitHub提交、实际CI和精确Git对象同步结果保存到本地发布回执，避免自引用。PR：https://github.com/zhoujinyi8/tokencomm/pull/2 。
''',encoding='utf-8')
sources=['bootstrap_cheng_native.py','fetch_cheng_dependencies.py','build_cheng_extensions.py','fetch_cheng_weights.py','cheng_stream_container.py','test_cheng_stream_container.py','cheng_runtime.py','smoke_cheng_backend.py','prepare_stage65_plan.py','run_stage65_cheng_diagnostic.py','decode_cheng_container.py','audit_stage65_cheng_diagnostic.py','cheng_packet_budget.py','test_cheng_packet_budget.py','audit_stage65_packet_budget.py','imagenet100_manifest.py','test_imagenet100_manifest.py','verify_imagenet100_manifest.py','audit_stage65_protocol_gaps.py','prepare_stage65_gradient_plan.py','run_stage65_gradient_smoke.py','audit_stage65_gradient_smoke.py','prepare_stage65_rate_pilot.py','run_stage65_rate_pilot.py','audit_stage65_rate_pilot.py','record_stage65_contracts.py','plot_stage65_codec_rates.py','report_stage65_cheng.py']
evidence=['stage65_native_bootstrap.json','stage65_dependency_wheels.json','stage65_native_build.json','stage65_backend_smoke.json','stage65_stock_weights.json','stage65_attn_download_verification.json','stage65_weight_download_deviation.json','stage65_container_contract.json','stage65_cheng_diagnostic_plan.json','stage65_cheng_diagnostic_lock.json','stage65_cheng_progress.json','stage65_cheng_audit.json','stage65_fresh_model0.json','stage65_fresh_model1.json','stage65_cheng_packet_budget.json','stage65_gradient_smoke_plan.json','stage65_gradient_smoke.json','stage65_gradient_smoke_audit.json','stage65_rate_pilot_plan.json','stage65_rate_pilot.json','stage65_rate_pilot_audit.json','stage65_remote_source_check.json','stage65_protocol_gaps.json','stage65_contracts.json']
entries={f'code/{n}':(r/n).read_bytes() for n in sources};entries.update({f'evidence/{n}':(run/n).read_bytes() for n in evidence})
entries['reports/'+report.name]=report.read_bytes();entries.update({'reports/'+p.name:p.read_bytes() for p in figures})
for p in (run/'stage65_cheng_cases').glob('*'):
    if p.suffix in ('.json','.ch65'):entries['diagnostic_streams/'+p.name]=p.read_bytes()
for p in (run/'stage65_rate_calibration').glob('*.ch65'):entries['proxy_calibration_streams/'+p.name]=p.read_bytes()
manifest={n:hashlib.sha256(v).hexdigest() for n,v in entries.items()};entries['SHA256.json']=json.dumps(manifest,indent=2).encode()
archive=out/(prefix+'_代码与证据.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for n,v in entries.items():z.writestr(n,v)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
print(json.dumps(dict(report=str(report),archive=str(archive),archive_sha256=sha(archive),cases=20,pilot_updates=pilot['updates_completed'],pilot_complete=pilot['complete'],entries=len(entries))))
