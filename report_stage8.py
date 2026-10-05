"""Generate Stage-8 distinct-dual-backbone TaiChi report and figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; RUN=ROOT/'runs/stage8_dual_taichi'; OUT=ROOT/'runs/stage8_report'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Noto Sans SC','SimHei','DejaVu Sans'],'axes.unicode_minus':False})


def mean_result(results,kind,snr,variant):
    xs=[x for x in results if x['channel']==kind and x['snr_db']==snr and x['variant']==variant]
    return {'accuracy':float(np.mean([x['exact_match'] for x in xs])),'ce':float(np.mean([x['ce'] for x in xs])),
            'seeds':[x['seed'] for x in xs],'accuracies':[x['exact_match'] for x in xs]}


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def main():
    status=json.loads((RUN/'status.json').read_text()); assert status['completed'] and not status['test_accessed']
    cfg=json.loads((RUN/'config.json').read_text()); result=json.loads((RUN/'results.json').read_text())
    history=[json.loads(x) for x in (RUN/'history.jsonl').read_text().splitlines() if x]
    checks=[x for x in history if 'validation' in x]
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    ax[0].plot([x['step'] for x in history],[x['loss'] for x in history],alpha=.75,color='#4c72b0'); ax[0].set(title='训练回答交叉熵',xlabel='训练步数',ylabel='交叉熵'); ax[0].grid(alpha=.2)
    ax[1].plot([x['step'] for x in checks],[100*x['validation']['exact_match'] for x in checks],'o-',color='#c44e52'); ax[1].set(title='AWGN 10 dB 验证准确率',xlabel='训练步数',ylabel='完整回答准确率（%）',ylim=(0,100)); ax[1].grid(alpha=.2)
    fig.suptitle('Stage-8 双独立视觉主干 TaiChi 训练',fontsize=15); fig.savefig(OUT/'training.png',dpi=170); plt.close(fig)
    rows=[]; compact=[]
    for kind,snr in [('awgn',10.),('awgn',120.),('rayleigh',10.)]:
        row=[]
        for variant in ['correct','wrong_class','zero']:
            m=mean_result(result['validation_ablations'],kind,snr,variant); row.append(m['accuracy']*100); compact.append({'channel':kind,'snr_db':snr,'variant':variant,**m})
        rows.append(row)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained'); x=np.arange(3); w=.25
    for j,(name,label,color) in enumerate([('correct','正确特征','#4c72b0'),('wrong_class','错配特征','#dd8452'),('zero','零特征','#55a868')]):
        vals=[next(c['accuracy'] for c in compact if c['channel']==k and c['snr_db']==s and c['variant']==name)*100 for k,s in [('awgn',10.),('awgn',120.),('rayleigh',10.)]]
        bars=ax.bar(x+(j-1)*w,vals,w,label=label,color=color); ax.bar_label(bars,fmt='%.1f',fontsize=9)
    ax.set(xticks=x,xticklabels=['AWGN 10 dB','AWGN 120 dB','Rayleigh 10 dB'],ylabel='完整回答准确率（%）',ylim=(0,105),title='150张验证图上的视觉特征消融（两个信道种子平均）'); ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'validation_controls.png',dpi=170); plt.close(fig)
    display=[]
    for i,(kind,snr) in enumerate([('AWGN','10'),('AWGN','120'),('Rayleigh','10')]): display.append([kind,snr,*[f'{v:.2f}%' for v in rows[i]]])
    report=f'''# 第八阶段：ViT 与 ConvNeXt 双独立视觉主干 TaiChi

## 阶段目标

此前 TaiChi 使用同一个 ResNet18 的双分辨率特征。本阶段改为两个结构独立的公开预训练视觉主干：ViT-B/16 提供 Transformer patch token，ConvNeXt-Tiny 提供卷积特征；两路经过独立适配器后进入 BAN、KAN、语义信道和回答解码器。这使视觉侧更接近论文的双主干设计。报告只使用 Stage-7 的 1,000 张训练图和 150 张验证图，新的 150 张测试图保持未读。

## 权重与特征身份

- ViT-B/16 IMAGENET1K_V1 SHA256：`{cfg['identity']['vit_weight_sha256']}`。
- ConvNeXt-Tiny IMAGENET1K_V1 SHA256：`{cfg['identity']['conv_weight_sha256']}`。
- 数据清单 SHA256：`{cfg['identity']['manifest_sha256']}`。
- 缓存包含 train/val 共 1,150 张，`test_opened=false`；训练状态 `test_accessed=false`。
- ViT 去除 CLS 后将 14×14 patch token 汇聚为 768×4×4；ConvNeXt 使用最终 768×7×7 特征。

## 训练配置

模型从 Stage-5 最佳 TaiChi 检查点继承语言、融合和通信权重；两个新适配器重新初始化。前 {cfg['alignment_steps']} 步在 AWGN 120 dB 下训练适配器、BAN 与 KAN，随后联合训练至 {cfg['steps']} 步并在 0–15 dB 随机采样。每步按类别均衡抽取 {cfg['batch_size']} 张图。模型含 {cfg['trainable_parameters']:,} 个可训练参数，按固定 AWGN 10 dB 验证完整回答准确率优先、交叉熵次优选择检查点，最佳步为 {result['best_step']}。

![训练曲线](runs/stage8_report/training.png)

## 验证集结果

{table(['信道','SNR / dB','正确特征','错配特征','零特征'],display)}

![验证消融](runs/stage8_report/validation_controls.png)

AWGN 10 dB 正确特征平均准确率为 {rows[0][0]:.2f}%，错配为 {rows[0][1]:.2f}%，零特征为 {rows[0][2]:.2f}%；Rayleigh 10 dB 正确特征平均为 {rows[2][0]:.2f}%。每个均值来自同一批 150 张验证图上的两个信道种子，不是独立测试，也不是 300 张不同图。

## 解释范围

双独立主干明显提高了当前受控任务的验证准确率，并通过错配/零特征对照证明回答依赖视觉表示。它仍使用 ImageNet 分类预训练而非论文的完整 CLIP 对齐权重，语言模块仍是小型回答解码器。Imagenette 来自 ImageNet，因此不能声称视觉主干在预训练阶段未见过该分布。Stage-7 测试集继续锁定，后续应先冻结 Stage-8 配置再一次性测试，或继续采用 Qwen-LoRA 大语言模型路线并另建开发/测试协议。
'''
    (ROOT/'STAGE8_REPORT.md').write_text(report,encoding='utf-8'); (OUT/'summary.json').write_text(json.dumps({'best_step':result['best_step'],'metrics':compact,'test_accessed':False},indent=2),encoding='utf-8')
    print(json.dumps({'best_step':result['best_step'],'awgn10':rows[0][0],'rayleigh10':rows[2][0],'test_accessed':False}))


if __name__=='__main__': main()
