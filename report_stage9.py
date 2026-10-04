"""Generate the Stage-9 Qwen2.5-LoRA large-model report after a completed run."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Noto Sans SC','SimHei','DejaVu Sans'],'axes.unicode_minus':False})


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/stage9_qwen_dual_taichi_balanced')
    parser.add_argument('--out',default='runs/stage9_report')
    args=parser.parse_args(); run=ROOT/args.run; out=ROOT/args.out
    status=json.loads((run/'status.json').read_text()); assert status['completed'] and not status['test_accessed']
    cfg=json.loads((run/'config.json').read_text()); results=json.loads((run/'results.json').read_text())
    raw_history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines() if x]
    # A clean resume can repeat uncheckpointed rows written after the previous
    # validation checkpoint. Keep the latest deterministic row per step.
    by_step={x['step']:x for x in raw_history}; history=[by_step[k] for k in sorted(by_step)]
    checks=[x for x in history if 'validation' in x]
    out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    ax[0].plot([x['step'] for x in history],[x['loss'] for x in history],color='#4c72b0',alpha=.8); ax[0].set(title='Qwen-LoRA 训练损失',xlabel='优化器步数',ylabel='回答损失'); ax[0].grid(alpha=.2)
    ax[1].plot([x['step'] for x in checks],[x['validation']['loss'] for x in checks],'o-',color='#c44e52',label='验证损失')
    ax2=ax[1].twinx(); ax2.plot([x['step'] for x in checks],[100*x['validation']['exact_match'] for x in checks],'s--',color='#55a868',label='完整回答准确率')
    ax[1].set(title='开发验证集检查点',xlabel='优化器步数',ylabel='验证损失'); ax2.set_ylabel('完整回答准确率（%）'); ax[1].grid(alpha=.2)
    fig.suptitle('Stage-9 双视觉主干与 Qwen2.5-0.5B LoRA 训练',fontsize=15); fig.savefig(out/'training.png',dpi=170); plt.close(fig)
    controls={x['variant']:x for x in results['validation_controls']}; order=['correct','wrong_class','zero']; names=['正确特征','错配特征','零特征']
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained'); vals=[100*controls[x]['exact_match'] for x in order]
    bars=ax.bar(names,vals,color=['#4c72b0','#dd8452','#999999']); ax.bar_label(bars,fmt='%.1f%%'); ax.set(ylabel='完整回答准确率（%）',ylim=(0,105),title=f"最佳检查点 {results['best_step']} 步的视觉消融"); ax.grid(axis='y',alpha=.2); fig.savefig(out/'controls.png',dpi=170); plt.close(fig)
    params=cfg['parameters']; rows=[[names[i],f"{100*controls[v]['exact_match']:.2f}%",f"{controls[v]['loss']:.4f}",controls[v]['count']] for i,v in enumerate(order)]
    checkpoint_rows=[[x['step'],f"{x['validation']['loss']:.4f}",f"{100*x['validation']['exact_match']:.2f}%",x['validation']['count']] for x in checks]
    report=f'''# 第九阶段：双视觉主干与 Qwen2.5 大模型语义通信

## 实现范围

本阶段将 Stage-8 的冻结 ViT-B/16 与 ConvNeXt-Tiny 特征、BAN、KAN 和语义信道接入 Qwen2.5-0.5B-Instruct。视觉融合得到16个前缀token，映射到Qwen的896维隐空间，经64维语义信道编码、AWGN传输与解码后，与问题token共同驱动自回归回答。Qwen基础权重冻结，只在注意力q/k/v/o投影中训练rank={cfg['args']['lora_rank']}的LoRA，同时训练视觉适配与信道模块。

总参数为 {params['total_parameters']:,}，可训练参数为 {params['trainable_parameters']:,}，占 {100*params['trainable_fraction']:.3f}%。微批为 {cfg['args']['batch_size']}，梯度累积 {cfg['args']['accumulate']} 次，有效批量 {cfg['effective_batch_size']}；共训练 {cfg['args']['steps']} 个优化器步。检查点只保存可训练参数、优化器、调度器和随机数状态，不重复保存冻结Qwen权重。

训练从旧运行的第100步检查点继续，但旧运行的64张验证图按类别排序截断，存在明显类别偏置；因此恢复时清空了旧最佳判据。第200、300、400步均改用固定十类别均衡子集完成模型选择。受本机CPU资源限制，恢复段使用微批2、梯度累积2（有效批量4）；该配置变更已写入检查点配置，不能与论文多卡大批量训练直接等同。

![训练曲线](runs/stage9_report/training.png)

## 开发验证结果

最佳检查点为第 **{results['best_step']}** 步。下表使用固定的 {cfg['args']['eval_n']} 张开发验证图和AWGN 10 dB条件。缩减验证子集按类别确定性抽取，覆盖全部10类，各类样本数之差不超过1；旧运行中按文件顺序截取前64张造成类别偏置的第100步结果不参与模型选择。

{table(['检查点','验证损失','完整回答准确率','图像数'],checkpoint_rows)}

{table(['视觉条件','完整回答准确率','回答损失','图像数'],rows)}

![视觉消融](runs/stage9_report/controls.png)

模型选择使用信道随机种子9900；最终消融使用独立种子9902，所以第400步模型选择准确率48.44%与正确特征消融的56.25%不同。错配特征准确率为0%，零特征为10.94%，说明生成结果主要由收到的视觉语义决定，而不是只凭固定问题模板或语言先验猜测类别。

## 数据与解释边界

训练使用1,000张Stage-7训练图，模型选择和消融只使用验证图。双主干缓存明确记录`test_opened=false`，训练记录`test_accessed=false`；150张Stage-7测试图未用于本阶段。Qwen权重来自公开本地检查点，视觉权重来自ImageNet分类预训练；该实现比小型回答解码器更接近论文的大语言模型路线，但仍不是论文原始CLIP/Gemma训练配方，也没有达到多机A800规模。验证子集参与模型选择，因此其指标不是最终测试成绩。
'''
    (ROOT/'STAGE9_REPORT.md').write_text(report,encoding='utf-8'); (out/'summary.json').write_text(json.dumps({'best_step':results['best_step'],'parameters':params,'controls':controls,'test_accessed':False},indent=2),encoding='utf-8')
    print(json.dumps({'best_step':results['best_step'],'correct':controls['correct']['exact_match'],'test_accessed':False}))


if __name__=='__main__': main()
