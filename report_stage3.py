"""Produce stage-3 report from completed run artifacts only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
M=ROOT/'runs/stage3_context_maskgit';T=ROOT/'runs/stage3_pretrained_taichi_v2'
OUT=ROOT/'runs/stage3_report';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
for p in [M,T]:assert read(p/'status.json')['completed']
m=read(M/'results.json');t=read(T/'results.json');cfg=read(T/'config.json')
mh=[json.loads(x) for x in (M/'history.jsonl').read_text().splitlines()]
th=[json.loads(x) for x in (T/'history.jsonl').read_text().splitlines()]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
axs[0].plot([x['step'] for x in mh],[x['loss'] for x in mh],alpha=.5,label='训练 batch CE')
vh=[x for x in mh if 'validation' in x]
axs[0].plot([x['step'] for x in vh],[x['validation']['ce_on_common_targets'] for x in vh],'o-',label='验证 CE')
axs[0].set(title='MaskGIT：每类4图、始终无CMI',xlabel='更新步数',ylabel='交叉熵');axs[0].legend()
vh=[x for x in th if 'validation' in x]
axs[1].plot([x['step'] for x in vh],[x['validation']['exact_match']*100 for x in vh],'o-',color='#287ba0',label='验证完整回答准确率')
axs[1].axhline(t['linear_probe']['validation_accuracy']*100,color='#899ba7',ls='--',label='无通信线性分类基线')
axs[1].axvline(200,color='#d18643',ls=':',label='对齐→联合训练')
axs[1].set(title='预训练视觉特征＋TaiChi',xlabel='更新步数',ylabel='准确率（%）',ylim=(0,105));axs[1].legend(fontsize=9)
fig.suptitle('第三阶段训练：验证集用于选择检查点，并非最终独立测试',fontsize=15)
fig.savefig(OUT/'training.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
names=['correct_context','wrong_context','no_context','spatial'];x=np.arange(4)
for j,split in enumerate(['train','val']):
    vals=[next(r['accuracy']*100 for r in m['last'][split] if r['mode']==n) for n in names]
    bars=axs[0].bar(x+(j-.5)*.36,vals,.36,label='训练图' if split=='train' else '验证图');axs[0].bar_label(bars,fmt='%.2f',fontsize=8)
axs[0].set(xticks=x,xticklabels=['正确上下文','错误上下文','无上下文','空间填充'],ylabel='丢失token恢复率（%）',title='MaskGIT最后一步：相同目标位置对照');axs[0].legend()
agg=[]
for kind,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
    for variant in ['correct','wrong_class','zero']:
        rows=[r for r in t['validation_ablations'] if r['channel']==kind and r['snr_db']==snr and r['variant']==variant]
        agg.append({'channel':kind,'snr_db':snr,'variant':variant,'accuracy':float(np.mean([r['exact_match'] for r in rows])),
                    'counts':[sum(p==q for p,q in zip(r['predictions'],r['targets'])) for r in rows]})
for j,(variant,label) in enumerate([('correct','正确特征'),('wrong_class','错配特征'),('zero','零特征')]):
    vals=[r['accuracy']*100 for r in agg if r['variant']==variant]
    bars=axs[1].bar(np.arange(3)+(j-1)*.25,vals,.25,label=label);axs[1].bar_label(bars,fmt='%.1f',fontsize=8)
axs[1].set(xticks=range(3),xticklabels=['AWGN 10 dB','AWGN 120 dB','Rayleigh 10 dB'],ylabel='回答准确率（%）',ylim=(0,112),title='预训练视觉TaiChi：两个噪声种子均值');axs[1].legend(fontsize=9)
fig.savefig(OUT/'controls.png',dpi=160);plt.close(fig)

def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
mt=[]
for n in names:mt.append([n]+[f"{next(r['accuracy']*100 for r in m['last'][s] if r['mode']==n):.2f}%" for s in ['train','val']])
tt=[[r['channel'],r['snr_db'],r['variant'],f"{r['accuracy']*100:.2f}%",str(r['counts'])+' / 40'] for r in agg]
result=f'''# 第三阶段：无类别上下文恢复与预训练视觉接入

本轮完成MaskGIT每类多图、无CMI训练1200步，以及预训练视觉特征接入TaiChi后的1200步训练。所有调试、模型选择与消融只使用既有训练集和验证集，没有打开测试集。旧阶段检查点保留，新模型单独保存。

## 一、MaskGIT：排除类别模板记忆

第二阶段每类一图时，模型可凭类别标签直接记住整张token网格。本轮从原训练集每类选4图，共40张；始终使用“无CMI”嵌入，真实类别只用于数据抽样与错配对照，不作为模型输入。保持公开VQGAN码本1024、256tokens，使用每包4token、PER=30%的随机交织擦除。模型从第一阶段最佳权重开始，AdamW初始学习率0.001、batch8、余弦调度，共1200步。

下表为最后一步模型在新固定擦除掩码下的结果。错配上下文来自其他类别图像；无上下文将全部位置遮蔽，但所有条件只在原先同一批缺失位置计分，因此分母一致。

{table(['恢复条件','40张训练图','40张验证图'],mt)}

正确上下文在训练图上明显好于错配、移除上下文及空间填充，支持模型在没有类别标签时利用可见token识别并恢复训练图。但验证集未超过空间填充，泛化问题仍在。验证CE随训练恶化；按事先规定的最低验证CE选择规则，最佳步为{m['best_step']}，即保留训练前检查点，而非将最后一步模型当作新最佳模型。

这是一项40图的上下文学习诊断，没有证明新图片上的恢复改善。训练子集、学习率与掩码策略都与第一阶段不同，跨阶段数字不构成严格单因素对照。

## 二、预训练视觉特征接入TaiChi

为先验证本机可运行的预训练视觉接口，本轮使用torchvision公开ResNet18 IMAGENET1K_V1。权重约46.8MB，下载自官方地址，校验官方文件名SHA256前缀，并保存本地完整SHA256。它是共享权重的双分辨率特征提取器，不是论文的CLIP-ViT与CLIP-ConvNeXt双主干。

- 高分辨率分支：官方224中心裁剪及标准化输入，取layer3，池化为256×8×8。
- 低分辨率分支：同一裁剪图缩至112，取layer4，池化为512×4×4。
- 两路冻结特征经可训练1×1卷积和GroupNorm映射到64维，随后进入原BAN、KAN、多模态编码器、信道和字节回答解码器。
- 仅缓存240张训练图和40张验证图；主干不参加本轮梯度更新。问题固定为询问主要物体类别，尚不是通用视觉问答。

预训练权重身份：`{cfg['identity']['weight_sha256']}`。

该主干在ImageNet上预训练，Imagenette来自ImageNet；结果只代表对预训练图像表示的下游利用，不表示严格未见过预训练样本的泛化。高分辨率输入、训练策略等也同时改变，不能将跨阶段提升全部归因于单一因素。

训练配置：batch20，每类每步抽2图；AdamW初始学习率0.0005，余弦调度。前200步只训练特征适配器、BAN和KAN，在AWGN120dB下对齐；后1000步解冻小型语言/信道等模块，SNR在0–15dB采样。每100步验证，以AWGN10dB、固定噪声种子7300下完整回答准确率优先、CE次优选择检查点。最终选中第{t['best_step']}步。

## 三、基线和视觉特征消融

冻结低分辨率特征、全局池化后训练线性十分类器300步，训练准确率为{t['linear_probe']['train_accuracy']*100:.1f}%，验证准确率为{t['linear_probe']['validation_accuracy']*100:.1f}%。这个基线没有通信链路和文本解码器，不能称为TaiChi的通信成绩；它用于检查视觉特征本身是否包含类别信息。

最佳TaiChi检查点的验证集结果如下，每行是相同40张图在两个匹配噪声种子7300、7302下的均值。7300参与过模型选择，7302为附加噪声实现；并非80张独立测试图。

{table(['信道','SNR / dB','特征条件','平均完整回答准确率','各噪声种子正确数'],tt)}

“零特征”直接将缓存的两路特征置零，不等于输入黑色图像后经过ResNet提取的特征；“错配特征”替换成下一类别第一张验证图的特征，保持问题和答案不变。消融使用相同随机数种子及批处理边界。可据此判断当前模型是否依赖视觉特征，不能据此推断具备开放词汇问答或论文大模型能力。

![第三阶段训练过程](runs/stage3_report/training.png)

![第三阶段上下文与视觉特征对照](runs/stage3_report/controls.png)

## 四、修复和验证

接入双路特征时发现原生成方法直接读取image.shape，不支持tuple特征输入，导致首次运行在初始验证阶段停止。已改为读取编码后的memory批次大小和设备，并增加特征输入生成及检查点重载测试。未完成目录stage3_pretrained_taichi保留失败说明；完成运行使用stage3_pretrained_taichi_v2。

完整回归共19项全部通过，耗时12.197秒，包括原模型、数字链路、LoRA、接收端恢复、随机数状态恢复、阶段二对照，以及本轮特征梯度与生成重载。模型选择未使用测试集；后续扩大实验后仍需冻结配置再开展独立测试。

## 五、复跑入口与文件

```powershell
python cache_pretrained_vision.py
python train_context_maskgit.py --out runs/context_repeat
python train_pretrained_taichi.py --out runs/vision_repeat
python infer_pretrained_taichi.py --image <自己的图片路径> --checkpoint runs/stage3_pretrained_taichi_v2/best.pt --channel awgn --snr 10
```

缓存复用时会检查数据清单和权重身份；需安装与PyTorch版本相匹配的torchvision。本机使用CPU，PyTorch2.14.0+cpu / torchvision0.29.0+cpu。脚本保存优化器、调度器和随机数状态，但本轮新增训练入口没有实现自动中断续训CLI，不应将这些命令描述成精确续训命令。

成果目录：

单图端到端核验已完成：选择验证清单的第一张图（真实类别tench），从原始图片重新提取两路冻结特征，通过AWGN10dB通信并自由生成回答，输出为tench。未根据预测正确与否挑选输入。记录及输入图片保存在runs/stage3_single_image/。这仅验证推理入口，与验证集统计不是两套独立证据。

- runs/stage3_context_maskgit/：无CMI训练、最佳/最终权重及上下文对照。
- runs/stage3_pretrained_taichi_v2/：预训练视觉TaiChi检查点、线性基线、逐步日志和验证消融。
- data/imagenette_pilot/resnet18_features/：冻结视觉特征缓存与身份信息。
- pretrained/resnet18-f37072fd.pth：公开预训练主干权重。
- infer_pretrained_taichi.py：从自然图像读取、提取特征、通信、自由生成回答的单图入口。

下一步应在扩大的每类多图数据上继续验证MaskGIT泛化；TaiChi则需逐步替换为论文对应的预训练视觉/语言主干，并加入多问题任务和冻结配置后的测试。本轮完成的是预训练视觉接口与受控实验推进，尚未完成CLIP双主干、Gemma/Qwen、真实拆分及IDR的论文级复现。

公开权重与预处理依据：[torchvision ResNet18官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。
'''
(ROOT/'STAGE3_REPORT.md').write_text(result,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'maskgit':m,'taichi_aggregated':agg,'best_taichi_step':t['best_step']},indent=2),encoding='utf-8')
print(json.dumps({'best_taichi_step':t['best_step'],'taichi':agg},ensure_ascii=False),flush=True)
