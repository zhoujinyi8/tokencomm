"""Label-side-information controls and figures for completed stage-2 diagnostics."""
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from real_maskgit import load_real_maskgit
from models import erase_packets

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/stage2_diagnostics'
torch.set_num_threads(4)
assert json.loads((OUT/'status.json').read_text())['completed']
config=json.loads((OUT/'config.json').read_text())
data=torch.load(ROOT/'data/imagenette_pilot/token_cache/train.pt',weights_only=True)
ids=data['ids'][config['train_indices']]; labels=data['labels'][config['train_indices']]
model=load_real_maskgit(torch.load(OUT/'maskgit_overfit.pt',map_location='cpu',weights_only=True)).eval()
rows=[]
with torch.inference_mode():
    for per in [.3,1.]:
        torch.manual_seed(5001); mask=erase_packets(ids,per,4)
        for name,lab in [('correct_cmi',labels),('no_cmi',torch.full_like(labels,model.classes)),('wrong_cmi',(labels+1)%10)]:
            recovered=model.recover(ids.masked_fill(mask,model.codes),mask,lab)
            assert torch.equal(recovered[~mask],ids[~mask])
            rows.append({'per':per,'condition':name,'missing_tokens':int(mask.sum()),
                         'accuracy':float((recovered[mask]==ids[mask]).float().mean())})
(OUT/'maskgit_condition_controls.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
ta=json.loads((OUT/'taichi_overfit_result.json').read_text())
ma=json.loads((OUT/'maskgit_overfit_result.json').read_text())
before=json.loads((OUT/'taichi_ablation_before.json').read_text())
hist={n:[json.loads(l) for l in (OUT/(n+'_overfit_history.jsonl')).read_text().splitlines()] for n in ['taichi','maskgit']}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
for ax,n,title in zip(axs[:2],['taichi','maskgit'],['TaiChi：10 张训练图拟合','MaskGIT：10 张训练图拟合']):
    h=hist[n]; ax.plot([x['step'] for x in h],[x['loss'] for x in h],color='#267ca2')
    ax.set(xlabel='优化器更新步',ylabel='交叉熵',title=title); ax.grid(alpha=.2)
    if n=='taichi': ax.set_yscale('log')
vals=[next(r['exact_match']*100 for r in ta['after_train_ablations'] if r['channel']=='awgn' and r['snr_db']==10 and r['image_variant']==v) for v in ['correct','wrong_class','zero']]
b=axs[2].bar(['正确图像','错配图像','零图像'],vals,color=['#267ca2','#d67555','#8d9aa2'])
axs[2].bar_label(b,fmt='%.1f%%',padding=3); axs[2].set(ylim=(0,115),ylabel='完整回答准确率（%）',title='TaiChi 拟合集图像消融：AWGN 10 dB')
fig.suptitle('第二阶段：训练集拟合与输入利用诊断（不是泛化成绩）',fontsize=16)
fig.savefig(OUT/'diagnostic_curves.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for j,(name,label) in enumerate([('correct_cmi','正确类别'),('no_cmi','无类别'),('wrong_cmi','错误类别')]):
    vals=[r['accuracy']*100 for r in rows if r['condition']==name]
    bars=ax.bar(np.arange(2)+(j-1)*.23,vals,.23,label=label); ax.bar_label(bars,fmt='%.1f',padding=3)
ax.set(xticks=[0,1],xticklabels=['PER=30%','全部 token 擦除'],ylim=(0,115),ylabel='丢失 token 精确恢复率（%）',title='MaskGIT 拟合集：类别条件依赖检查')
ax.legend(); fig.savefig(OUT/'condition_controls.png',dpi=150); plt.close(fig)

def tab(headers,rs):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rs)
pre=[]
for snr in [10.,120.]:
    for variant in ['correct','wrong_class','zero']:
        rs=[r for r in before if r['channel']=='awgn' and r['snr_db']==snr and r['image_variant']==variant]
        pre.append([snr,variant,f"{np.mean([r['exact_match'] for r in rs])*100:.2f}%",f"{np.mean([r['ce'] for r in rs]):.4f}"])
after=[[r['snr_db'],r['image_variant'],f"{r['exact_match']*100:.1f}%"] for r in ta['after_train_ablations'] if r['channel']=='awgn']
control=[[r['per'],r['condition'],f"{r['accuracy']*100:.2f}%"] for r in rows]
report=f'''# 第二阶段：输入利用与小样本拟合诊断

本阶段执行了原模型验证集消融、梯度检查，以及两个模型各 {config['steps']} 步的小样本拟合实验。实验未读取测试集，未替换第一阶段的最佳模型。新权重均标记为 diagnostic_only，不应当作最终模型部署或论文性能结果。

## 1. 设置与公平比较

从既有240张训练图中取每类第一张，共10张，采用全批量、AdamW、学习率0.001、无权重衰减做记忆能力检查。TaiChi保留信道投影和归一化，在AWGN 120 dB下训练；MaskGIT使用每包4个token、PER=30%的每步随机交织擦除与正确类别条件。两者从第一阶段各自最佳检查点开始。

TaiChi原模型消融使用40张验证图、两个匹配噪声种子（2040、2041）。保持问题和答案不变，将图像替换为下一类别第一张或全零图。同一条件下使用匹配随机数以减少信道噪声干扰。错配方案是跨类别替换，不是无放回随机排列。120 dB为近乎无噪声对照，不等于数值严格零噪声。

代码中的channel=none会绕过信道编码器和解码器，因此其0%回答正确率仅说明绕过已训练投影会改变内部表示，不能用来得出“去掉噪声反而变差”的结论。更合适的去噪声对照是保留投影的高SNR条件。

## 2. 第一阶段检查点：验证集图像消融

{tab(['SNR / dB','图像条件','两个噪声种子的平均准确率','平均回答CE'],pre)}

这些百分比是同一40张图在两个噪声种子下的平均值，不代表80张独立图像。正确图像的结果仍接近10%类别基线；高SNR也没有明显解决分类问题，因此无法将主要误差简单归因于信道噪声。零图像CE低于正确图像CE，进一步说明回答字节交叉熵不能单独代表图像识别能力。

图像输入以及lr_patch、lr_vit、hr_cnn、BAN、KAN、多模态编码器、信道投影和回答解码器均检测到非零梯度，记录见gradient_diagnostics.json。该结果排除了本次样本上完全断开的梯度路径，但并不证明各模块都学到了有效表示。

## 3. TaiChi 10图拟合与图像依赖

{tab(['SNR / dB','图像条件','拟合图片完整回答准确率'],after)}

拟合前，高SNR下这10张训练图的准确率为{ta['initial_train']['exact_match']*100:.1f}%。拟合后正确图像可达到100%，错配图像降至0%，零图像为10%。这支持该结构能利用图像信息记忆小样本映射，之前的低成绩不能简单解释为图像完全没有进入模型。

但是，同一拟合检查点在40张验证图上高SNR准确率只有{ta['validation_after_high_snr']['exact_match']*100:.1f}%。这说明小样本记忆成功并未转化为泛化能力；此权重不应取代第一阶段模型。

![训练曲线及TaiChi输入消融](runs/stage2_diagnostics/diagnostic_curves.png)

## 4. MaskGIT 10图拟合与类别记忆

拟合前，固定训练图擦除模式下迭代恢复准确率为{ma['initial_train']['iterative_accuracy']*100:.2f}%。拟合后，三个掩码种子下的训练图恢复准确率分别为{', '.join(f"{r['iterative_accuracy']*100:.2f}%" for r in ma['final_train_new_masks'])}。其中3001为训练中定期观察的诊断掩码，3002和3003为训练结束后增加的掩码；均为已训练过的图片，不是独立图像测试。

40张验证图的丢失token迭代恢复准确率为{ma['validation_iterative_accuracy']*100:.2f}%，按各批丢失token数加权。该验证擦除采用单独种子；不能与第一阶段不同掩码的测试值直接作性能提升结论。

{tab(['设定PER','类别条件','拟合图丢失token恢复率'],control)}

![MaskGIT类别条件依赖](runs/stage2_diagnostics/condition_controls.png)

每类仅一张训练图时，类别标签与图片一一对应。本次即使全部token擦除，正确类别条件下也达到100%恢复；30%擦除下去掉类别条件降至2.24%，错误类别降至0%。这确认模型能够通过“类别→整张token网格”的模板记忆完成该拟合任务，训练图上的高恢复率不能独立证明上下文恢复能力。后续应采用每类多图和无类别条件的拟合实验进一步区分能力。

## 5. 当前结论与下一步

本阶段将“训练损失下降但准确率低”的问题推进到了可检查的机制诊断：视觉梯度链路存在，小型TaiChi可以利用图像记忆训练样本，但泛化仍差；MaskGIT需要区分类别模板记忆与上下文恢复。当前结果支持继续做数据规模和预训练主干实验，不支持宣布论文性能复现完成。

后续优先扩展每类样本并接入预训练表示，保留空间/频率和零图像基线；验证集用于调试，最终测试需冻结配置后评估。仅在10张样本上继续延长训练不应作为性能提升方案。

## 6. 文件与重跑

诊断代码：diagnose_stage2.py；类别对照与绘图：report_stage2.py。
结果目录：runs/stage2_diagnostics/，包括初始消融、梯度、两份800步日志、两个诊断检查点、训练后评价和完成状态。新增test_stage2_diagnostics.py的2项检查已通过：跨类别替换保持目标不变；匹配噪声评价可复现且不改变外部随机数状态。两份新增主脚本亦通过语法编译检查。本轮未修改核心模型，也未重复宣称运行此前15项完整回归。

```powershell
python diagnose_stage2.py --out runs/stage2_diagnostics_repeat --steps 800
```

report_stage2.py默认读取本轮固定结果目录。全程CPU 4线程；具体随机种子、训练图片索引与原检查点步数见config.json。本次增加的是诊断实验，并未引入论文级CLIP/Gemma/Qwen或作者MaskGIT权重。
'''
(ROOT/'STAGE2_REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps({'controls':rows,'report':str(ROOT/'STAGE2_REPORT.md')},ensure_ascii=False),flush=True)
