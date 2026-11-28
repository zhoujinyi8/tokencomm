"""Render the frozen stage-4 final-test report without rerunning evaluation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;RUN=ROOT/'runs/stage4_final_test'
r=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
assert json.loads((RUN/'status.json').read_text())['completed']

def tgroup(channel,snr,variant):
    rows=[x for x in r['taichi'] if x['channel']==channel and x['snr_db']==snr and x['variant']==variant]
    return {'mean':float(np.mean([x['accuracy'] for x in rows])),'min':min(x['accuracy'] for x in rows),
            'max':max(x['accuracy'] for x in rows),'per_seed':[x['accuracy'] for x in rows],
            'mean_ce':float(np.mean([x['response_ce'] for x in rows]))}
ta=[]
for channel,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
    for variant in ['correct','wrong_class','zero']:ta.append({'channel':channel,'snr_db':snr,'variant':variant,**tgroup(channel,snr,variant)})

def mgroup(key,per):
    rows=[x for x in r[key] if x['per']==per]
    total=sum(x['missing_tokens'] for x in rows)
    return {'weighted_maskgit':sum(x['maskgit_correct'] for x in rows)/total,
            'weighted_spatial':sum(x['spatial_correct'] for x in rows)/total,
            'per_seed_maskgit':[x['maskgit_accuracy'] for x in rows],'missing_tokens_total':total}
ms=[]
for key in ['maskgit_selected','maskgit_last_diagnostic']:
    for per in [.1,.3,.5]:ms.append({'model':key,'per':per,**mgroup(key,per)})

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for j,(variant,label) in enumerate([('correct','正确特征'),('wrong_class','错配特征'),('zero','零特征')]):
    vals=[next(x['mean'] for x in ta if x['channel']==c and x['snr_db']==s and x['variant']==variant)*100 for c,s in [('awgn',10.),('awgn',120.),('rayleigh',10.)]]
    bars=axs[0].bar(np.arange(3)+(j-1)*.25,vals,.25,label=label);axs[0].bar_label(bars,fmt='%.1f',fontsize=8)
axs[0].set(xticks=range(3),xticklabels=['AWGN 10 dB','AWGN 120 dB','Rayleigh 10 dB'],ylabel='完整回答准确率（%）',ylim=(0,105),title='TaiChi：40张独立测试图，5个信道种子均值');axs[0].legend(fontsize=9)
for j,(key,label) in enumerate([('maskgit_selected','验证选择模型'),('maskgit_last_diagnostic','训练最后模型')]):
    vals=[next(x['weighted_maskgit'] for x in ms if x['model']==key and x['per']==p)*100 for p in [.1,.3,.5]]
    bars=axs[1].bar(np.arange(3)+(j-.5)*.28,vals,.28,label=label);axs[1].bar_label(bars,fmt='%.2f',fontsize=8)
sp=[next(x['weighted_spatial'] for x in ms if x['model']=='maskgit_selected' and x['per']==p)*100 for p in [.1,.3,.5]]
axs[1].plot(range(3),sp,'o--',color='#555',label='空间填充')
axs[1].set(xticks=range(3),xticklabels=['PER=10%','PER=30%','PER=50%'],ylabel='丢失token精确恢复率（%）',title='MaskGIT：40张独立测试图，5个新掩码');axs[1].legend(fontsize=9)
fig.suptitle('第四阶段：冻结检查点的一次性独立测试',fontsize=15)
fig.savefig(RUN/'final_test.png',dpi=160);plt.close(fig)

def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,x))+' |\n' for x in rows)
trows=[]
for x in ta:trows.append([x['channel'],x['snr_db'],x['variant'],f"{x['mean']*100:.2f}%",f"{x['min']*100:.1f}%–{x['max']*100:.1f}%",f"{x['mean_ce']:.4f}"])
mrows=[]
for x in ms:mrows.append(['验证选择模型' if x['model']=='maskgit_selected' else '最后训练模型（诊断）',f"{x['per']*100:.0f}%",f"{x['weighted_maskgit']*100:.3f}%",f"{x['weighted_spatial']*100:.3f}%",x['missing_tokens_total']])
report=f'''# 第四阶段：冻结模型的独立测试报告

本阶段首次开启此前保留的40张测试图。测试协议在计算结果前写入`protocol.json`，第三阶段检查点已经冻结；本轮没有训练、选择检查点或根据测试结果修改模型。所有检查点记录SHA256，便于确认被评估的确切文件。

## 1. TaiChi独立测试

{table(['信道','SNR / dB','视觉特征条件','5种子平均准确率','种子范围','平均回答CE'],trows)}

AWGN 10 dB下，正确视觉特征平均准确率为{tgroup('awgn',10.,'correct')['mean']*100:.2f}%，五个种子的范围为{tgroup('awgn',10.,'correct')['min']*100:.1f}%–{tgroup('awgn',10.,'correct')['max']*100:.1f}%；Rayleigh 10 dB平均为{tgroup('rayleigh',10.,'correct')['mean']*100:.2f}%。这低于第三阶段验证集结果，但明显高于正确特征被错配或清零后的对照。

每个种子都作用于相同40张测试图，所以“5个种子”是信道噪声重复，不是200张独立图片。均值和范围只描述信道随机性的敏感程度，不提供样本独立的置信区间。AWGN 120 dB保留信道投影和归一化，可作为近乎无噪声对照。

ResNet18在ImageNet上预训练，而Imagenette是ImageNet子集；本测试独立于当前下游训练和模型选择，但不满足“预训练阶段完全未见”的严格条件。成绩不能直接外推到开放词汇或通用视觉问答。

## 2. MaskGIT独立测试

{table(['模型','设定PER','MaskGIT恢复率','空间填充','5次丢失token总数'],mrows)}

“验证选择模型”是第三阶段按最低验证CE选出的检查点，其最佳步为{r['maskgit_selected_step']}，即无CMI多图训练前的模型；“最后训练模型”仅作为预先记录的诊断结果，不改变模型选择。最后模型虽然在40张训练图上达到较高恢复率，但测试集仍未超过空间填充，说明训练上下文记忆没有转化成新图像恢复能力。

恢复率只统计缺失位置。五个掩码种子在同一40张图上重复采样，token之间也具有空间和包级相关性，因此没有把全部token当作独立样本计算置信区间。

## 3. 结果图

![冻结模型独立测试结果](runs/stage4_final_test/final_test.png)

左图显示TaiChi对正确视觉特征的依赖；右图比较验证选择模型、最后训练模型和空间填充。图中所有结果来自测试集，测试之后没有重新训练或改选检查点。

## 4. 可复现信息

- TaiChi检查点：`{r['checkpoint_sha256']['taichi']}`，选择步数{r['taichi_best_step']}。
- MaskGIT验证选择检查点：`{r['checkpoint_sha256']['maskgit_selected']}`，步数{r['maskgit_selected_step']}。
- MaskGIT最后训练检查点：`{r['checkpoint_sha256']['maskgit_last_diagnostic']}`，步数{r['maskgit_last_step']}。
- 完整逐图预测、每种子回答CE和每掩码计数：`runs/stage4_final_test/results.json`。
- 测试前固定的条件与随机种子：`runs/stage4_final_test/protocol.json`。
- 测试特征缓存：`runs/stage4_final_test/test_features.pt`，仅由冻结官方ResNet18提取。

复跑命令：

```powershell
python evaluate_stage4_final.py --out runs/stage4_final_test_repeat
python report_stage4.py
```

第二个命令默认读取本轮固定目录；重复执行第一条会得到相同固定种子结果，但它不产生新的独立测试集。后续若继续调参，应建立新的未见数据，而不能继续将这40张测试图当作最终盲测集。

## 5. 阶段结论

预训练视觉特征使小型TaiChi在冻结后的测试集上获得了明显高于十分类基线的结果，并且错配特征对照确认回答依赖正确图像信息。这是当前最实质性的性能推进，但仍不是论文CLIP双主干、Gemma/Qwen或通用VQA复现。

MaskGIT方面，扩大到每类多图并移除CMI后，训练图上的上下文利用能力得到验证，但独立测试仍弱于空间填充。继续推进时应扩大数据并引入预训练上下文生成模型；本测试集已用于评估，后续模型需要另建最终留出集。
'''
(ROOT/'STAGE4_REPORT.md').write_text(report,encoding='utf-8')
(RUN/'summary.json').write_text(json.dumps({'taichi':ta,'maskgit':ms},indent=2),encoding='utf-8')
print(json.dumps({'taichi':ta,'maskgit':ms},ensure_ascii=False),flush=True)
