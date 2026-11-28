"""Render complete clean-codec evidence and exact physical transport coverage without paper-wide claims."""
import base64,hashlib,json,zipfile
from pathlib import Path
from datetime import datetime,timezone
from plot_stage66_cheng import plot
r=Path(__file__).resolve().parent;run=r/'runs';out=r.parents[2]/'outputs';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
s=read('stage66_session_audit.json');assert s['all_checks_passed'] and s['holdout_cases']==100 and s['grid_updates']==300
assert s['script_sha256']==sha(r/'audit_stage66_session.py')
for n,h in s['source_receipt_sha256'].items():assert sha(run/n)==h
ta=read('stage66_transport_audit.json');resource=read('stage66_resource_budget.json');hp=read('stage66_holdout_plan.json')
prefix='TokCom_阶段66Cheng低码率校准与留出验证_2026-11-28';figures=plot(r,out,prefix)
table=['| 损失权重 lambda | 开发集实际熵流 bpp | 含34字节头 bpp | PSNR/dB | LPIPS ↓ | CLIP ↑ | 进入预设窗口 |',
       '|---:|---:|---:|---:|---:|---:|:---:|']
for row in s['grid_selection']['rows']:
    name=row['arm_id'];m=s['grid_metric_means'][name]
    table.append(f"| {int(name.split('_')[1])/100000:.5f} | {row['mean_entropy_payload_bpp']:.6f} | {row['mean_total_stored_bpp']:.6f} | {m['psnr_decoded']:.3f} | {m['lpips_normalized']:.6f} | {m['clip_clipped']:.6f} | {'是（冻结候选）' if row['within_rate_window'] else '否'} |")
h=s['holdout_means'];dev=resource['groups'][s['selected_arm_id']]['means'];held=resource['groups']['frozen_holdout']['means']
transport_lines=[]
if ta['complete']:
    transport_lines=['| SNR/dB | 单发字节完全正确帧/50 | CRC重传完整且正确帧/50 | CRC错误接受包 | CRC平均前向QAM符号 | 真值参考平均前向QAM符号 |',
                     '|---:|---:|---:|---:|---:|---:|']
    for snr,policies in ta['complete_snr_groups'].items():
        a=policies['single_pass_padded'];b=policies['receiver_crc_arq_padded'];c=policies['sender_oracle_arq_padded']
        transport_lines.append(f"| {snr} | {a['container_exact_frames']} | {b['complete_and_exact_frames']} | {b['accepted_crc_payload_errors']} | {b['mean_forward_qam_symbols']:.2f} | {c['mean_forward_qam_symbols']:.2f} |")
else:
    transport_lines=[f"当前逐SNR覆盖为 `{ta['frames_per_snr']}`。不完整组不发布成功率或均值；精确整数账本保存在审计JSON中。"]
endpoint_text=''
if ta['complete']:
    ep=read('stage66_transport_endpoints_summary.json')
    crc=read('stage66_crc_endpoint_completion.json');corrupt=read('stage66_crc_miss_metrics.json')
    endpoint_text=f"字节一致的{ep['reference_rows']}条端点引用已审计的同检查点、同源图干净重建，不计作新模型推理。随后对CRC重传唯一字节错误帧作实际隔离解码及全指标补测：PSNR={corrupt['psnr_decoded']:.3f}dB、LPIPS={corrupt['lpips_normalized']:.6f}、CLIP={corrupt['clip_clipped']:.6f}。因此，工程CRC重传的300条端点现由299条精确干净引用和1条实际受损重建组成，未删除失败帧。6dB完整50帧均值为PSNR={crc['per_snr']['6.0']['means']['psnr_decoded']:.3f}dB、LPIPS={crc['per_snr']['6.0']['means']['lpips_normalized']:.6f}、CLIP={crc['per_snr']['6.0']['means']['clip_clipped']:.6f}；其余SNR全部字节正确。单发仍有292帧缺完整重建指标，不能用成功子集代表全组。详见 `stage66_crc_endpoint_completion.json`。"
report=out/(prefix+'.md')
report.write_text(f'''# 阶段66：Cheng低码率开发校准、固定留出验证与实际字节传输

记录时间：{datetime.now(timezone.utc).isoformat()}。本轮从阶段65已审计的0.031396 bpp试点继续，另建新计划；未改写此前实验结果。

## 三设置固定预算校准

三个Cheng Anchor设置分别从同一公开q1权重开始，使用同一100张训练图、图像顺序、随机种子和100次更新预算。主学习率0.0001、辅助学习率0.001、batch=1、梯度上限1；损失权重为0.00015、0.00025、0.00040。它们是工程选择，尚未证实为作者配方。训练图按原580条验证/测试记录的文件、像素及源ID排除重叠。

训练前固定规则：三个设置及各十张开发图必须全部完成；只在纯熵流平均码率进入0.036–0.044 bpp后，按距0.04的绝对差选择，精确相等时按冻结顺序；不根据PSNR、LPIPS、CLIP或留出结果选模型。该±10%容差是本工程诊断规则。

{chr(10).join(table)}

共300次更新、30次真实开发图编码、30次新进程解码复核均完成；复核像素摘要与编码阶段一致。全部三个设置另测同一规范的感知指标，保留全部结果。

![开发码率与质量]({prefix}_开发码率与质量.svg)

## 冻结候选后的100张十类留出图

选择 `{s['selected_arm_id']}` 后，冻结检查点摘要 `{s['selected_checkpoint_sha256']}`，再读取原阶段44的100张留出图作本项独立干净编解码评估。这是十类、每类十张的既有留出集，不是论文ImageNet100百类数据集。训练图、开发校准图与本组按冻结身份排除重叠；不重新抽图、训练或选检查点。

| 固定留出评估，100/100张 | 均值 |
|---|---:|
| 实际完整熵流 bpp | {h['entropy_payload_bpp']:.6f} |
| 含自定义34字节头 bpp | {h['total_stored_bpp']:.6f} |
| 似然估计 bpp | {h['estimated_likelihood_bpp']:.6f} |
| 实际重建 PSNR/dB | {h['psnr_decoded']:.3f} |
| LPIPS（normalize=True） | {h['lpips_normalized']:.6f} |
| CLIP（ViT-B/32、PIL裁剪预处理） | {h['clip_clipped']:.6f} |

实际留出熵流码率比0.04高{s['holdout_entropy_rate_relative_difference_from_target']*100:.2f}%，仍在开发诊断窗口内。开发集均值为0.037451 bpp，留出集观测均值更高；没有利用此结果再调参。似然估计不能替代实际字节码率。全部100张的像素、字节长度和PSNR经独立审计；预先固定索引0、24、49、74、99由新进程重新解码，像素最大差均为0。每图标量结果保存在 `stage66_session_audit.json`。

## 完整容器和首发资源

| 同样40bit负载、CRC11、半率卷积码、每次26个QAM符号 | 开发10图 | 留出100图 |
|---|---:|---:|
| 纯熵流、所有元数据另传的反事实首发QAM符号 | {dev['entropy_only_metadata_out_of_band']['forward_qam_symbols']:.2f} | {held['entropy_only_metadata_out_of_band']['forward_qam_symbols']:.2f} |
| 实际完整CH65容器首发QAM符号 | {dev['with_custom_header']['forward_qam_symbols']:.2f} | {held['with_custom_header']['forward_qam_symbols']:.2f} |
| 完整容器相对TokCom 1664符号分帧参考 | {dev['with_custom_header']['ratio_to_tokcom_1664_symbol_first_pass']:.4f}倍 | {held['with_custom_header']['ratio_to_tokcom_1664_symbol_first_pass']:.4f}倍 |

CH65的34字节头由本工程定义，不是已经确认的论文报头。完整容器账本包含此头、分包尾部补零、CRC、卷积编码和QAM补齐。纯熵流一行假设形状、两个流的边界等信息另行传送，不能视为免费可部署协议。接收端共享检查点及长度约定、分包控制、反馈和CMI的完整空口开销仍未计齐。即使熵流接近0.04 bpp，也不能据此声称资源完全相等。

![留出码率与首发开销]({prefix}_留出码率与首发开销.svg)

## 独立物理字节传输研究

另立冻结计划：同一候选的十张开发图完整容器，经连续5字节/40bit分包；6个SNR、每图5次接收，共300帧。CRC11、[101,111]卷积码及16-QAM采用现有Sionna 0.19.2/TensorFlow 2.15.1后端。每包/每次尝试使用固定SeedSequence坐标产生独立AWGN。三种停止规则共享同次噪声：单发原始接收字节、接收端首次CRC通过即停止、发送端真实负载相等的理想参考。最多32次，保留截尾和CRC错误接受，不利用真值延长CRC重传。

两个无噪分帧诊断通过；24帧先导逐次轨迹审计通过后，按相同计划继续。本轮当前审计覆盖 **{ta['completed_frames']}/300帧**，完整SNR组 **{len(ta['complete_snr_groups'])}/6**；先导24帧摘要保持一致。

{chr(10).join(transport_lines)}

十位分组只是传输适配器，不是语义token；不将其错误数称为TokCom TER。真值重传为理想参考，CRC重传不是无条件可靠保证。完整结果及整数账本见 `stage66_transport_audit.json`。该代理模型、十类开发集和本工程报头尚不构成论文“Cheng”和“Cheng+R”质量曲线复现。

![实际字节传输]({prefix}_实际字节传输.svg)

{endpoint_text}

另立六坐标诊断（图0、trial0、全部六个SNR，不按结果挑选）：五个容器被格式/几何校验拒绝，一个受损码流解码成功但PSNR仅13.166dB；另有上述CRC漏检帧的事后机制诊断。这两组机制诊断的原生解码在固定256×256、latent 4×4、至多65536输入字节的独立子进程中运行，60秒超时则终止自有进程树。只作进程/几何/时间隔离，不宣称内存沙箱；不把拒绝或失败帧的PSNR填成零或干净值。

CRC漏检感知补测初次因2/4解码线程数不同，触发严格像素摘要门槛。保留初次源代码和锁；控制重复表明固定2线程三次像素完全一致，4线程最大差约1.97×10⁻⁶。修正补测解码线程数后通过同一摘要门槛，感知模型仍按原4线程规范计算；没有放宽像素验收。详见 `stage66_crc_metric_preflight_deviation.json`。

## 论文和代码版本核对

本轮下载并固定[arXiv v4源归档](https://arxiv.org/src/2502.12096v4)及[v5源归档](https://arxiv.org/src/2502.12096v5)的完整SHA256。主TeX只增加摘要末尾90字符的代码链接句；实验部分逐字一致，五个其他源成员相同；归档构建元数据另增加spec_version和TeX Live版本。此结论来自源文件比较，不等同于代码执行或检查点等价。

又核对用户原始2025期刊文档的已有文本提取缓存：百类数据集、0.039 bpp token表示、约0.04 bpp微调Cheng、16-QAM、0.05 MHz、7bit类别CMI及两组TER/SNR数字与arXiv吻合。该项没有重新核验原始PDF字节，也不宣称期刊全文逐字等价。

[论文第VI节](https://arxiv.org/html/2502.12096v5#S6)给出的约0.04 bpp微调要求仍缺精确变体、权重、训练数据和配方。作者公开HEAD仍为0422ed32dcde3347e0bb6697f343f7a9ff7b3214；本轮只读检查[公开问题#1](https://github.com/liqiao19/TokenCom_Code/issues/1)未见评论。未发送外部消息。

## 校验、文件和下一步

此前30份冻结历史证据保持摘要一致，包括原主协议与主汇总、固定/随机分包、输入约定和完整匹配帧对照；上轮检查点保持一致。新增17项纯协议检查（选择门槛9、字节适配与噪声8）通过，与原43项合计60项加入CI。模型、原生后端及实际信道运行是本地资产验证，不把CI未执行的11项集成测试列为通过。

代码、不可变计划、审计和逐图标量随Git快照发布；权重、数据图片、重建数组、编译产物及论文全文不上传Git。证据ZIP另含本项码流和物理轨迹，可用于本地逐次停止规则复核。定点重建PNG仅为本地诊断图，索引在留出运行前固定，不按质量挑图。Git提交及本次CI回执在发布后另存，避免自引用。

下一步：在已有隔离机制上固定单发受损熵流的完整统计与失败处理规则，补齐缺失图像端点；继续追查作者百类图像身份、检查点和完整反馈/控制/CMI资源协议。300帧物理传输和300条工程CRC重传端点本轮已完整，不需重复运行。本轮没有根据留出集调整检查点。**论文完整复现尚未完成。**
''',encoding='utf-8')
import markdown
body=markdown.markdown(report.read_text(encoding='utf-8'),extensions=['tables','fenced_code'])
for figure in figures:
    if figure.suffix=='.svg':
        encoded=base64.b64encode(figure.read_bytes()).decode('ascii');body=body.replace('src="'+figure.name+'"','src="data:image/svg+xml;base64,'+encoded+'"')
qualitative=next(p for p in figures if p.name.endswith('_定点重建.png'))
body+='<h2>固定索引的本地重建诊断</h2><p>五个索引在运行前固定；本图只随本地报告和证据包提供。</p><img alt="原图与固定检查点重建" src="data:image/png;base64,'+base64.b64encode(qualitative.read_bytes()).decode('ascii')+'">'
html=out/(prefix+'.html')
html.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TokCom阶段66复现进度</title><style>body{max-width:1120px;margin:36px auto;padding:0 24px 60px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#243244;background:#fdfdfc}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:36px;border-bottom:2px solid #d7e4e9;padding-bottom:8px}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:14px;margin:20px 0;width:100%}th,td{padding:9px 12px;border:1px solid #dbe1e6;white-space:nowrap}th{background:#edf4f6}img{max-width:100%;height:auto}code{overflow-wrap:anywhere;background:#f0f3f5;padding:2px 4px;font-size:13px}a{color:#126f98}p{margin:14px 0}</style><main>'+body+'</main></html>',encoding='utf-8')
sources=sorted(set(r.glob('*stage66*.py'))|{r/n for n in ('cheng_rate_selection.py','test_cheng_rate_selection.py','cheng_byte_transport.py','test_cheng_byte_transport.py','cheng_runtime.py','cheng_stream_container.py','cheng_packet_budget.py','packet_arq_accounting.py','smoke_stage25_sionna_link.py')})
entries={'code/'+p.name:p.read_bytes() for p in sources}
for n in list(s['source_receipt_sha256'])+['stage66_session_audit.json']:entries['evidence/'+n]=(run/n).read_bytes()
entries['reports/'+report.name]=report.read_bytes()
entries['reports/'+html.name]=html.read_bytes()
if ta['complete']:entries['evidence/stage66_transport_endpoints.json']=(run/'stage66_transport_endpoints.json').read_bytes()
entries['evidence/stage66_received_headers.json']=(run/'stage66_received_headers.json').read_bytes()
entries['evidence/stage66_crc_miss_metrics_preflight_source.py']=(run/'stage66_crc_miss_metrics_preflight_source.py').read_bytes()
for folder in ('stage66_decode_probe','stage66_crc_miss_probe'):
    for path in (run/folder).iterdir():
        if path.suffix in ('.json','.ch65'):entries[folder+'/'+path.name]=path.read_bytes()
for p in figures:entries['reports/'+p.name]=p.read_bytes()
for folder in ('stage66_rate_calibration','stage66_holdout_cases','stage66_transport_cases'):
    for p in (run/folder).iterdir():
        if p.suffix in ('.json','.ch65') or (folder=='stage66_transport_cases' and p.suffix=='.npz'):entries[folder+'/'+p.name]=p.read_bytes()
manifest={n:hashlib.sha256(data).hexdigest() for n,data in entries.items()};entries['SHA256.json']=json.dumps(manifest,indent=2).encode()
archive=out/(prefix+'_代码与证据.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for n,data in entries.items():z.writestr(n,data)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps(dict(report=str(report),archive=str(archive),archive_sha256=sha(archive),entries=len(entries),holdout_cases=100,transport_frames=ta['completed_frames'])))
