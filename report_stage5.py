"""Create stage-5 validation report and plots; never opens the held-out test split."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/stage5_report';OUT.mkdir(exist_ok=True)
M=ROOT/'runs/stage5_context_maskgit';T=ROOT/'runs/stage5_pretrained_taichi';D=ROOT/'data/imagenette_stage5'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
for p in [M,T]:
    s=read(p/'status.json');assert s['completed'] and s['test_accessed'] is False
manifest=read(D/'manifest.json');token_summary=read(D/'token_cache/summary.json')
assert manifest['old_pilot_overlap']==0 and token_summary['test_opened'] is False
m=read(M/'results.json');t=read(T/'results.json')
mh=[json.loads(x) for x in (M/'history.jsonl').read_text().splitlines()]
th=[json.loads(x) for x in (T/'history.jsonl').read_text().splitlines()]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
mv=[x for x in mh if 'validation' in x]
axs[0].plot([x['step'] for x in mh],[x['loss'] for x in mh],alpha=.45,label='训练batch CE')
axs[0].plot([x['step'] for x in mv],[x['validation']['ce_on_common_targets'] for x in mv],'o-',label='验证CE')
axs[0].set(title='MaskGIT：800训练图、无CMI',xlabel='更新步数',ylabel='交叉熵');axs[0].legend()
tv=[x for x in th if 'validation' in x]
axs[1].plot([x['step'] for x in tv],[x['validation']['exact_match']*100 for x in tv],'o-',label='通信问答验证准确率')
axs[1].axhline(t['linear_probe']['validation_accuracy']*100,ls='--',color='#718895',label='冻结特征线性基线')
axs[1].axvline(200,ls=':',color='#d1803c',label='适配器对齐→联合训练')
axs[1].set(title='预训练视觉TaiChi：800训练图',xlabel='更新步数',ylabel='准确率（%）',ylim=(0,102));axs[1].legend(fontsize=9)
fig.suptitle('第五阶段：扩大且与旧pilot不重叠的数据；仅训练/验证',fontsize=15)
fig.savefig(OUT/'training.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
names=['correct_context','wrong_context','no_context','spatial'];labels=['正确上下文','错配上下文','无上下文','空间填充']
vals=[next(x['accuracy'] for x in m['best']['val'] if x['mode']==n)*100 for n in names]
b=axs[0].bar(labels,vals,color=['#287ba0','#db8454','#8f9ca5','#555']);axs[0].bar_label(b,fmt='%.2f',padding=3)
axs[0].set(ylabel='丢失token恢复率（%）',title='MaskGIT：100张验证图，固定新掩码')
groups=[]
for ch,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
    groups.append([np.mean([x['exact_match'] for x in t['validation_ablations'] if x['channel']==ch and x['snr_db']==snr and x['variant']==v])*100 for v in ['correct','wrong_class','zero']])
for j,label in enumerate(['正确特征','错配特征','零特征']):
    bars=axs[1].bar(np.arange(3)+(j-1)*.25,[x[j] for x in groups],.25,label=label);axs[1].bar_label(bars,fmt='%.1f',fontsize=8)
axs[1].set(xticks=range(3),xticklabels=['AWGN 10 dB','AWGN 120 dB','Rayleigh 10 dB'],ylabel='完整回答准确率（%）',ylim=(0,105),title='TaiChi：100张验证图，两个噪声种子均值');axs[1].legend(fontsize=9)
fig.savefig(OUT/'validation_controls.png',dpi=160);plt.close(fig)

def table(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,x))+' |\n' for x in rows)
mrows=[[labels[i],f'{vals[i]:.3f}%'] for i in range(4)]
trows=[]
for (ch,snr),g in zip([('AWGN',10),('AWGN',120),('Rayleigh',10)],groups):trows.append([ch,snr,*[f'{x:.1f}%' for x in g]])
report=f'''# 第五阶段：扩大数据训练与新留出集建设

本阶段建立了与旧pilot完全不重叠的新Imagenette子集：每类80张训练、10张验证、10张留出测试，共800/100/100张。旧pilot的240张训练图和80张验证/测试图分别从对应源划分排除；新划分同时检查文件SHA256与解码像素摘要，未发现跨划分重复。新的100张测试图尚未被训练、缓存或模型选择脚本读取。

## 1. 数据身份与边界

- 数据随机种子：{manifest['seed']}。
- 训练/验证/测试：{manifest['split_counts']['train']}/{manifest['split_counts']['val']}/{manifest['split_counts']['test']}。
- 旧pilot重叠：{manifest['old_pilot_overlap']}。
- VQ token缓存仅包含：{', '.join(token_summary['splits'])}；`test_opened={str(token_summary['test_opened']).lower()}`。
- ResNet特征缓存调用同样只指定train和val；训练配置均记录`test_accessed=false`。

数据源仍为Imagenette。ResNet18在ImageNet上预训练，因此TaiChi实验不满足预训练阶段未见数据的严格条件；本阶段的“不重叠”指与本项目旧pilot和当前下游训练/验证划分不重叠。

## 2. 预训练视觉TaiChi

模型从第三阶段第500步最佳检查点热启动，在800张新训练图上更新1600步。每步按类别均衡抽样20张；前200步在AWGN 120 dB下训练适配器、BAN和KAN，之后联合训练小型语言/信道模块并在0–15 dB采样SNR。验证模型选择仍以AWGN 10 dB完整回答准确率优先、回答CE次优。

最佳检查点为第{t['best_step']}步。冻结低分辨率特征的无通信线性分类基线在100张验证图上为{t['linear_probe']['validation_accuracy']*100:.1f}%。通信问答模型结果：

{table(['信道','SNR / dB','正确特征','错配特征','零特征'],trows)}

AWGN 10 dB下达到84%，比新验证集的10%类别基线高，且错配特征降至0%；Rayleigh 10 dB两个种子为76%和74%。结果表明扩大训练数据后，视觉特征经过BAN/KAN、信道压缩和文本解码仍能保留较强类别信息。但这些是用于模型选择和诊断的验证结果，尚不是新留出集成绩。

## 3. 无CMI MaskGIT

模型使用800张训练图、100张验证图，始终输入无类别条件；公开VQGAN冻结。训练2400步，batch8，PER=30%、每包4 token，验证每400步一次。验证CE持续下降，因此选择第{m['best_step']}步。

{table(['验证恢复条件','丢失token精确恢复率'],mrows)}

正确上下文明显好于错配和完全移除上下文，说明模型在新验证图片上开始利用已收到token，而不只是类别模板记忆。不过1.185%仍低于空间填充2.192%，未形成优于简单基线的恢复器。训练图正确上下文为{next(x['accuracy'] for x in m['best']['train'] if x['mode']=='correct_context')*100:.3f}%，也仅略低于对应空间填充，扩大数据抑制了第二阶段那种小样本记忆，但模型容量、训练目标或预训练程度仍不足。

## 4. 结果图

![第五阶段训练曲线](runs/stage5_report/training.png)

![第五阶段验证集对照](runs/stage5_report/validation_controls.png)

## 5. 文件与复跑

- `data/imagenette_stage5/`：新划分、清单、训练/验证token及视觉特征缓存。
- `runs/stage5_pretrained_taichi/`：TaiChi最佳/最后权重、线性基线和验证消融。
- `runs/stage5_context_maskgit/`：MaskGIT最佳/最后权重、训练日志与上下文对照。
- `prepare_stage5_imagenette.py`：排除旧pilot并建立新划分。

```powershell
python prepare_stage5_imagenette.py --train-parquet ../../work/downloads/imagenette_train.parquet --val-parquet ../../work/downloads/imagenette_validation.parquet --out data/imagenette_stage5
python cache_pretrained_vision.py --data data/imagenette_stage5 --splits train val
python cache_real_tokens.py --data data/imagenette_stage5 --splits train val
python train_pretrained_taichi.py --data data/imagenette_stage5/resnet18_features --init runs/stage3_pretrained_taichi_v2/best.pt --steps 1600 --out runs/stage5_pretrained_taichi
python train_context_maskgit.py --data data/imagenette_stage5/token_cache --per-class 80 --steps 2400 --eval-every 400 --out runs/stage5_context_maskgit
```

## 6. 下一步

当前配置应先冻结，再首次缓存和评估新的100张测试图。测试协议需要在打开测试前记录信道条件、随机种子、检查点SHA256和评价指标，测试后不再调参。若后续继续开发模型，应另建新的留出集。
'''
(ROOT/'STAGE5_REPORT.md').write_text(report,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'maskgit':m,'taichi':t,'test_opened':False},indent=2),encoding='utf-8')
print(json.dumps({'maskgit_best_step':m['best_step'],'taichi_best_step':t['best_step'],'test_opened':False},ensure_ascii=False))

if __name__=='__main__':pass
