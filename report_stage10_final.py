"""Create the preregistered Stage-10 final-test report and publication figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent; RUN=ROOT/'runs/stage10_final_test'; FIG=ROOT/'figures/stage10_final'
NAMES=['tench','English springer','cassette player','chain saw','church','French horn','garbage truck','gas pump','golf ball','parachute']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def main():
    status=json.loads((RUN/'status.json').read_text()); data=json.loads((RUN/'summary.json').read_text())
    assert status['completed'] and data['test_accessed'] and not data['used_for_training']
    rows=[json.loads(x) for x in (RUN/'results.jsonl').read_text().splitlines() if x]; FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
    for kind,label,color,marker in [('awgn','AWGN','#4c72b0','o'),('rayleigh','Rayleigh','#c44e52','s')]:
        g=[x for x in data['summary'] if x['kind']==kind]; snr=[x['snr_db'] for x in g]
        ax[0].errorbar(snr,[100*x['exact_match_mean'] for x in g],yerr=[100*x['exact_match_std'] for x in g],label=label,color=color,marker=marker,capsize=4,linewidth=2)
        ax[1].plot(snr,[x['loss_mean'] for x in g],label=label,color=color,marker=marker,linewidth=2)
    ax[0].set(title='预注册最终测试准确率',xlabel='SNR (dB)',ylabel='完整回答准确率（%）',ylim=(98.5,100.15),xticks=[0,5,10,20]); ax[0].grid(alpha=.25); ax[0].legend()
    ax[1].set(title='预注册最终测试损失',xlabel='SNR (dB)',ylabel='回答损失',xticks=[0,5,10,20]); ax[1].grid(alpha=.25); ax[1].legend()
    fig.suptitle('Stage-10 第2200步模型：150张独立测试图',fontsize=15); fig.savefig(FIG/'channel_sweep.png',dpi=180); plt.close(fig)
    controls={x['variant']:x for x in data['controls']}; order=['correct','wrong_class','zero']; labels=['正确特征','错配特征','零特征']
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained'); bars=ax.bar(labels,[100*controls[x]['exact_match'] for x in order],color=['#4c72b0','#dd8452','#999999'])
    ax.bar_label(bars,fmt='%.1f%%'); ax.set(title='最终测试视觉依赖消融（AWGN 10 dB）',ylabel='完整回答准确率（%）',ylim=(0,105)); ax.grid(axis='y',alpha=.2); fig.savefig(FIG/'controls.png',dpi=180); plt.close(fig)
    matrix=[]
    for kind in ('awgn','rayleigh'):
        g=[x for x in rows if x['variant']=='correct' and x['kind']==kind and x['seed'] in (4101,4102,4103)]
        matrix.append([100*np.mean([x['per_class_accuracy'][str(c)] for x in g]) for c in range(10)])
    fig,ax=plt.subplots(figsize=(12,3.2),layout='constrained'); im=ax.imshow(matrix,vmin=90,vmax=100,cmap='YlGn')
    ax.set_xticks(range(10),NAMES,rotation=25,ha='right'); ax.set_yticks([0,1],['AWGN','Rayleigh']); ax.set_title('跨全部SNR与种子的逐类别平均准确率（%）')
    for i in range(2):
        for j in range(10): ax.text(j,i,f'{matrix[i][j]:.1f}',ha='center',va='center',color='black')
    fig.colorbar(im,ax=ax,label='准确率（%）'); fig.savefig(FIG/'per_class.png',dpi=180); plt.close(fig)
    metric_rows=[[x['kind'].upper(),f"{x['snr_db']} dB",f"{100*x['exact_match_mean']:.2f}%",f"{100*x['exact_match_std']:.2f}%",f"{x['loss_mean']:.6f}",x['seeds']] for x in data['summary']]
    control_rows=[[labels[i],f"{controls[v]['loss']:.6f}",f"{100*controls[v]['exact_match']:.2f}%",controls[v]['count']] for i,v in enumerate(order)]
    report=f'''# Stage-10 预注册最终测试报告

## 协议与数据隔离

本报告严格使用提交在测试开启前的`STAGE10_FINAL_TEST_PROTOCOL.json`。协议SHA-256为`{data['protocol_sha256']}`；冻结模型为第{data['checkpoint_step']}步，测试集150张、每类15张。测试集已用于最终评价，但`used_for_training=false`，所有预注册条件均需完整报告，结果不得用于继续调参或更换模型。

## 信道结果

{table(['信道','SNR','准确率均值','总体标准差','平均损失','种子数'],metric_rows)}

![最终信道扫描](figures/stage10_final/channel_sweep.png)

## 视觉消融

{table(['条件','损失','完整回答准确率','图像数'],control_rows)}

![最终视觉消融](figures/stage10_final/controls.png)

## 逐类别稳定性

![逐类别准确率](figures/stage10_final/per_class.png)

跨条件错误主要集中在`golf ball`类别；其他类别在大多数条件下保持100%。这说明独立测试没有完全复制开发集的满分结果，同时仍保持很高的整体准确率。

## 结论边界

这是对当前独立实现的预注册测试，不是作者原始模型的精确回放。视觉主干使用ImageNet预训练，而Imagenette是ImageNet子集，因此结果包含预训练分布重叠优势。样本规模仅150张，每个信道条件三个种子；总体标准差描述信道种子波动，不是数据集抽样置信区间。测试集已经开启，后续任何新训练都必须使用新的留出集，不能再把本测试集用于模型选择。
'''
    (ROOT/'STAGE10_FINAL_TEST_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({'conditions':len(data['summary']),'controls':len(data['controls']),'test_images':data['test_images']}))


if __name__=='__main__': main()
