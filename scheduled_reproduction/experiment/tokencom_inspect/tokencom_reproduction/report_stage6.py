"""Summarize frozen stage-5 checkpoints evaluated once on the stage-5 holdout."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;RUN=ROOT/'runs/stage6_final_test';OUT=ROOT/'runs/stage6_report';OUT.mkdir(exist_ok=True)
r=json.loads((RUN/'results.json').read_text(encoding='utf-8'));assert json.loads((RUN/'status.json').read_text())['completed']
def tg(ch,snr,var):
    xs=[x for x in r['taichi'] if x['channel']==ch and x['snr_db']==snr and x['variant']==var]
    return {'mean':float(np.mean([x['accuracy'] for x in xs])),'min':min(x['accuracy'] for x in xs),'max':max(x['accuracy'] for x in xs),
            'counts':[x['correct'] for x in xs],'ce':float(np.mean([x['response_ce'] for x in xs]))}
ta=[{'channel':c,'snr_db':s,'variant':v,**tg(c,s,v)} for c,s in [('awgn',10.),('awgn',120.),('rayleigh',10.)] for v in ['correct','wrong_class','zero']]
def mg(per):
    xs=[x for x in r['maskgit'] if x['per']==per];n=sum(x['missing_tokens'] for x in xs)
    return {'per':per,'maskgit':sum(x['maskgit_correct'] for x in xs)/n,'spatial':sum(x['spatial_correct'] for x in xs)/n,'tokens':n,
            'per_seed':[x['maskgit_accuracy'] for x in xs]}
ms=[mg(p) for p in [.1,.3,.5]]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for j,(v,label) in enumerate([('correct','正确特征'),('wrong_class','错配特征'),('zero','零特征')]):
    vals=[next(x['mean'] for x in ta if x['channel']==c and x['snr_db']==s and x['variant']==v)*100 for c,s in [('awgn',10.),('awgn',120.),('rayleigh',10.)]]
    bars=axs[0].bar(np.arange(3)+(j-1)*.25,vals,.25,label=label);axs[0].bar_label(bars,fmt='%.1f',fontsize=8)
axs[0].set(xticks=range(3),xticklabels=['AWGN 10 dB','AWGN 120 dB','Rayleigh 10 dB'],ylabel='完整回答准确率（%）',ylim=(0,100),title='TaiChi：100张新留出图，5个信道种子');axs[0].legend(fontsize=9)
bars=axs[1].bar(np.arange(3)-.16,[x['maskgit']*100 for x in ms],.32,label='MaskGIT');axs[1].bar_label(bars,fmt='%.2f',fontsize=8)
axs[1].plot(np.arange(3),[x['spatial']*100 for x in ms],'o--',color='#555',label='空间填充')
axs[1].set(xticks=range(3),xticklabels=['PER=10%','PER=30%','PER=50%'],ylabel='丢失token精确恢复率（%）',title='MaskGIT：100张新留出图，5个掩码');axs[1].legend()
fig.suptitle('第六阶段：冻结模型的一次性新留出集测试',fontsize=15);fig.savefig(OUT/'final_test.png',dpi=160);plt.close(fig)
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,x))+' |\n' for x in rows)
trows=[[x['channel'],x['snr_db'],x['variant'],f"{x['mean']*100:.2f}%",f"{x['min']*100:.1f}%–{x['max']*100:.1f}%",str(x['counts'])] for x in ta]
mrows=[[f"{x['per']*100:.0f}%",f"{x['maskgit']*100:.3f}%",f"{x['spatial']*100:.3f}%",x['tokens']] for x in ms]
report=f'''# 第六阶段：扩大数据模型的独立留出测试

第五阶段在800/100训练验证图上冻结配置后，本阶段首次读取100张新的留出测试图。协议、随机种子、数据清单SHA256和检查点SHA256在读取图像前保存在`runs/stage6_final_test/protocol.json`。测试完成后没有训练、模型选择或参数修改。

## TaiChi

{table(['信道','SNR / dB','视觉条件','5种子平均','种子范围','每种子正确数 / 100'],trows)}

AWGN 10 dB下正确特征平均准确率为{tg('awgn',10.,'correct')['mean']*100:.2f}%，五个种子为{tg('awgn',10.,'correct')['counts']}；近无噪声AWGN为{tg('awgn',120.,'correct')['mean']*100:.2f}%；Rayleigh 10 dB为{tg('rayleigh',10.,'correct')['mean']*100:.2f}%。正确特征与错配/零特征的差异保留在新测试图上，支持模型使用视觉信息。五个种子使用同一100张图，表示信道随机性的重复，不是500张独立图。

## MaskGIT

{table(['设定PER','MaskGIT恢复率','空间填充','5次丢失token总数'],mrows)}

MaskGIT在新留出图上仍低于空间填充。该模型的正确上下文优势已在第五阶段验证集观察到，但独立测试没有显示超越空间基线的泛化恢复能力。

## 图表

![第六阶段最终测试](runs/stage6_report/final_test.png)

## 解释边界

测试图独立于本项目所有前序训练、验证和pilot划分；但ResNet18在ImageNet上预训练，而Imagenette来自ImageNet，因此TaiChi不能被表述为严格的预训练未见分布测试。MaskGIT使用公开ImageNet VQGAN也有相同背景限制。

这100张图现在已被用于最终评价。后续开发不应再将它们用于调参或再次汇报为盲测；新的MaskGIT架构需要使用新的训练/验证划分并另建留出集。

## 成果文件

- `runs/stage6_final_test/protocol.json`：测试前冻结协议与哈希。
- `runs/stage6_final_test/results.json`：逐种子回答、交叉熵和token恢复统计。
- `runs/stage6_final_test/test_cache.pt`：测试图的冻结特征和VQ token，只写在结果目录。
- `evaluate_stage6_final.py`：一次性测试脚本。
'''
(ROOT/'STAGE6_REPORT.md').write_text(report,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'taichi':ta,'maskgit':ms},indent=2),encoding='utf-8')
print(json.dumps({'awgn10':tg('awgn',10.,'correct'),'rayleigh10':tg('rayleigh',10.,'correct'),'maskgit':ms},ensure_ascii=False))
