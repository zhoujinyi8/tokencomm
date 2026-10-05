"""Plot and report the completed Stage-10 development channel sweep."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; RUN=ROOT/'runs/stage10_channel_sweep'; FIG=ROOT/'figures/stage10'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})


def main():
    data=json.loads((RUN/'summary.json').read_text()); assert data['test_accessed'] is False
    rows=data['summary']; FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    for kind,label,color,marker in [('awgn','AWGN','#4c72b0','o'),('rayleigh','Rayleigh','#c44e52','s')]:
        group=[r for r in rows if r['kind']==kind]; x=[r['snr_db'] for r in group]
        mean=[100*r['exact_match_mean'] for r in group]; std=[100*r['exact_match_std'] for r in group]
        ax[0].errorbar(x,mean,yerr=std,label=label,color=color,marker=marker,capsize=4,linewidth=2)
        ax[1].plot(x,[r['loss_mean'] for r in group],label=label,color=color,marker=marker,linewidth=2)
    ax[0].set(title='三种信道种子的完整回答准确率',xlabel='SNR (dB)',ylabel='准确率（%）',ylim=(95,100.4),xticks=[0,5,10,20]); ax[0].grid(alpha=.25); ax[0].legend()
    ax[1].set(title='三种信道种子的平均回答损失',xlabel='SNR (dB)',ylabel='回答损失',xticks=[0,5,10,20]); ax[1].grid(alpha=.25); ax[1].legend()
    fig.suptitle(f"Stage-10 最佳检查点 {data['checkpoint_step']}：开发验证信道扫描",fontsize=15)
    fig.savefig(FIG/'channel_sweep.png',dpi=180); plt.close(fig)
    table='| 信道 | SNR | 准确率均值 | 标准差 | 平均损失 | 种子数 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows: table+=f"| {r['kind'].upper()} | {r['snr_db']} dB | {100*r['exact_match_mean']:.2f}% | {100*r['exact_match_std']:.2f}% | {r['loss_mean']:.6f} | {r['seeds']} |\n"
    report=f'''# Stage-10 开发验证信道鲁棒性

使用第{data['checkpoint_step']}步最佳模型，在固定{data['validation_images']}张十类别均衡开发验证图上扫描AWGN与Rayleigh信道。每个SNR使用三个固定信道随机种子101、202、303。

{table}

![信道扫描](figures/stage10/channel_sweep.png)

## 解释边界

这些是开发验证结果，不是最终盲测。模型在AWGN随机SNR训练下获得高鲁棒性；Rayleigh没有作为本阶段主训练信道，因此0 dB下降到98%更能反映分布变化。视觉主干使用ImageNet预训练，而Imagenette属于ImageNet子集，结果包含预训练迁移优势。三种信道种子只能提供初步波动估计，不能替代大样本置信区间。
'''
    (ROOT/'STAGE10_CHANNEL_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'conditions':len(rows),'test_accessed':False,'figure':str(FIG/'channel_sweep.png')}))


if __name__=='__main__': main()
