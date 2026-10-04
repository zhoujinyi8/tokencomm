"""Generate Stage-10 strong-training plots and a concise evidence report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/stage10_qwen_strong'; OUT=ROOT/'runs/stage10_report'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def main():
    status=json.loads((RUN/'status.json').read_text()); assert status['completed'] and not status['test_accessed']
    cfg=json.loads((RUN/'config.json').read_text()); results=json.loads((RUN/'results.json').read_text())
    raw=[json.loads(x) for x in (RUN/'history.jsonl').read_text().splitlines() if x]
    # Interrupted work after a saved checkpoint may be recomputed. Latest row wins.
    unique={x['step']:x for x in raw}; hist=[unique[k] for k in sorted(unique)]; checks=[x for x in hist if 'validation' in x]
    OUT.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    ax[0].plot([x['step'] for x in hist],[x['loss'] for x in hist],color='#4c72b0',alpha=.75)
    ax[0].set(title='Stage-10 强化训练损失',xlabel='全局优化步数',ylabel='回答损失'); ax[0].set_yscale('log'); ax[0].grid(alpha=.2)
    steps=[x['step'] for x in checks]; losses=[x['validation']['loss'] for x in checks]; acc=[100*x['validation']['exact_match'] for x in checks]
    ax[1].plot(steps,losses,'o-',color='#c44e52'); ax2=ax[1].twinx(); ax2.plot(steps,acc,'s--',color='#55a868')
    ax[1].set(title='固定均衡开发验证',xlabel='全局优化步数',ylabel='验证损失'); ax2.set_ylabel('完整回答准确率（%）'); ax[1].grid(alpha=.2)
    fig.suptitle('8,000张训练图上的 Qwen2.5-LoRA 语义通信强化训练',fontsize=15); fig.savefig(OUT/'training.png',dpi=170); plt.close(fig)
    controls={x['variant']:x for x in results['validation_controls']}; order=['correct','wrong_class','zero']; names=['正确特征','错配特征','零特征']
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained'); vals=[100*controls[x]['exact_match'] for x in order]
    bars=ax.bar(names,vals,color=['#4c72b0','#dd8452','#999999']); ax.bar_label(bars,fmt='%.1f%%')
    ax.set(ylabel='完整回答准确率（%）',ylim=(0,105),title=f"Stage-10 最佳检查点 {results['best_step']} 步视觉消融"); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'controls.png',dpi=170); plt.close(fig)
    checkpoint_rows=[[x['step'],f"{x['validation']['loss']:.6f}",f"{100*x['validation']['exact_match']:.2f}%",x['validation']['count']] for x in checks]
    control_rows=[[names[i],f"{controls[v]['loss']:.6f}",f"{100*controls[v]['exact_match']:.2f}%",controls[v]['count']] for i,v in enumerate(order)]
    report=f'''# Stage-10：8,000张图的Qwen双视觉语义通信强化训练

## 配置

训练从Stage-9第400步权重开始，在8,000张十类均衡Imagenette训练图上继续到第2,400步。微批{cfg['args']['batch_size']}、梯度累积{cfg['args']['accumulate']}，有效批量{cfg['effective_batch_size']}。模型共{cfg['parameters']['total_parameters']:,}个参数，其中{cfg['parameters']['trainable_parameters']:,}个可训练。测试集保持封存。

![训练与验证曲线](figures/stage10/training.png)

## 固定开发验证

{table(['步数','验证损失','完整回答准确率','图像数'],checkpoint_rows)}

## 最佳模型视觉消融

{table(['条件','损失','完整回答准确率','图像数'],control_rows)}

![视觉消融](figures/stage10/controls.png)

## 解释边界

这些结果来自参与模型选择的开发验证子集，不是最终盲测。ViT-B/16与ConvNeXt-Tiny来自ImageNet预训练，而Imagenette是ImageNet子集，因此不能宣称视觉主干从未见过相同分布。当前实现是对论文结构的独立近似复现，不是作者训练状态的精确回放；最终测试集仅在模型、信道条件和评价协议冻结后开启。
'''
    (ROOT/'STAGE10_REPORT.md').write_text(report,encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'best_step':results['best_step'],'checkpoints':checks,'controls':controls,'test_accessed':False},indent=2),encoding='utf-8')
    print(json.dumps({'best_step':results['best_step'],'checks':len(checks),'correct':controls['correct']['exact_match']}))


if __name__=='__main__': main()
