"""Create the detailed development report for the locked-test ContextMaskGIT v2 run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Noto Sans SC', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / 'stage7_report'


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' +
            ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows))


def rows(values):
    return {x['mode']: x for x in values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/stage7_context_maskgit_v2_b16_fixed', help='completed Stage-7 run directory')
    args = parser.parse_args()
    RUN = ROOT / args.run
    status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    if not status['completed'] or status['test_accessed']:
        raise RuntimeError('Stage-7 run must complete while preserving the test lock.')
    config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
    before = json.loads((RUN / 'before.json').read_text(encoding='utf-8'))
    results = json.loads((RUN / 'results.json').read_text(encoding='utf-8'))
    history = [json.loads(line) for line in (RUN / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line]
    OUT.mkdir(parents=True, exist_ok=True)

    checkpoints = [x for x in history if 'validation' in x]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    axes[0].plot([x['step'] for x in history], [x['loss'] for x in history], color='#4271ae', alpha=.8, label='训练掩码交叉熵')
    axes[0].set(xlabel='训练步数', ylabel='交叉熵', title='训练目标')
    axes[0].grid(alpha=.2)
    axes[1].plot([x['step'] for x in checkpoints], [x['validation']['ce'] for x in checkpoints], 'o-', color='#c44e52', label='验证掩码交叉熵')
    axes[1].axvline(results['best_step'], ls='--', color='#555', label=f"选择检查点：{results['best_step']}步")
    axes[1].set(xlabel='训练步数', ylabel='交叉熵', title='检查点选择（仅验证集）')
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=9)
    fig.suptitle('第七阶段：ContextMaskGIT v2 训练曲线', fontsize=15)
    fig.savefig(OUT / 'training.png', dpi=170); plt.close(fig)

    bval, aval = rows(before['val']), rows(results['best']['val'])
    order = ['correct', 'wrong', 'none', 'spatial']
    cn = {'correct': '正确上下文', 'wrong': '错配上下文', 'none': '无上下文', 'spatial': '空间填补'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    axes[0].bar(range(4), [bval[x]['accuracy'] * 100 for x in order], color='#aaa')
    axes[0].set(xticks=range(4), xticklabels=[cn[x] for x in order], ylabel='丢失 token 精确恢复率（%）', title='训练前：验证集对照')
    axes[1].bar(range(4), [aval[x]['accuracy'] * 100 for x in order], color=['#4c72b0', '#dd8452', '#999999', '#777777'])
    axes[1].set(xticks=range(4), xticklabels=[cn[x] for x in order], ylabel='丢失 token 精确恢复率（%）', title='最佳检查点：验证集对照')
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('同一验证掩码下的上下文消融，PER=30%、每包4 token', fontsize=14)
    fig.savefig(OUT / 'validation_controls.png', dpi=170); plt.close(fig)

    control_rows = []
    for split in ['train', 'val']:
        best = rows(results['best'][split])
        control_rows += [[split, cn[key], f"{best[key]['accuracy'] * 100:.3f}%", best[key]['missing_tokens'], (f"{best[key]['ce']:.4f}" if 'ce' in best[key] else '—')]
                         for key in order]
    val_gain = (aval['correct']['accuracy'] - aval['wrong']['accuracy']) * 100
    spatial_gap = (aval['correct']['accuracy'] - aval['spatial']['accuracy']) * 100
    report = f'''# 第七阶段：ContextMaskGIT v2 的独立开发集训练与验证

## 本阶段目的

第六阶段的 100 张最终测试图已完成一次性评价，因此不再可用于任何结构修改、训练或模型选择。本阶段为新版 ContextMaskGIT v2 新建第三套 Imagenette 划分，以检验二维位置、局部图像先验与全局 token 上下文是否能改善 VQ token 的丢失恢复。报告中的所有数值来自训练集或验证集，**不包含新的 150 张最终测试图**。

## 数据与隔离协议

Stage-7 从公开 Imagenette 原始 Parquet 文件构建，每类采样 100/15/15 张，合计训练 1,000 张、验证 150 张、测试 150 张。划分排除了 pilot 和 Stage-5 的已使用图片；下载文件哈希、源图片 SHA256 与解码 RGB 像素哈希均记录在 `data/imagenette_stage7/manifest.json`。跨 split 的文件和像素重复数均为零。

公开 VQGAN 的预处理为 `PIL ImageOps.fit(256, LANCZOS) → RGB float[0,1]`，checkpoint SHA256 为 `{config['identity']['tokenizer_sha256']}`。训练缓存含 1,150 张 train/val 图，缓存元数据为 `test_opened: false`；测试目录下没有 token 缓存。本次模型选择只以验证掩码交叉熵决定。

## 模型与训练

ContextMaskGIT v2 不输入类别、问答答案或文本描述。输入是 16×16 的 1,024 类 VQ token 网格；已丢失 token 替换为 mask code。模型将独立的行和列嵌入相加，以局部深度可分离卷积块建模邻域，再以 Transformer 建模全局关联。接收端在 {config['args']['steps']} 步训练中随机产生 30% 的、每包 4 token 的丢包，并采用 {config['args']['d']} 维隐层、{config['args']['layers']} 个 Transformer 层和 {config['args']['local_blocks']} 个局部块。恢复采用 8 次置信度逐步填补，所有未丢失 token 原样保留。

![训练曲线](runs/stage7_report/training.png)

最佳验证检查点为第 **{results['best_step']}** 步。`best.pt`、`last.pt`、完整逐步训练记录和模型配置均位于 `{args.run}/`。

## 验证集上下文对照

下表使用固定随机种子、同一批丢失位置和 PER=30% 的验证条件。正确上下文从同一幅图的接收 token 取得；错配上下文改用不同类别图像；无上下文将输入全部遮蔽；空间填补是非学习的邻近复制基线。精确恢复率仅在原始丢失位置统计。

{table(['数据','条件','精确恢复率','统计的丢失 token','掩码交叉熵'], control_rows)}

![验证对照](runs/stage7_report/validation_controls.png)

验证集上正确上下文相对错配上下文的差值为 **{val_gain:.3f} 个百分点**；相对空间填补的差值为 **{spatial_gap:.3f} 个百分点**。该差值用于检查模型是否利用同图的可接收 token，而不是只学习码本的静态频率。该观察仅适用于这个开发验证集，不能解释为最终留出测试结论。

## 结果的范围与下一步

本阶段已经完成独立开发数据上的结构训练和验证。150 张 Stage-7 测试图仍保持未读；只有在确认本结构、训练步数、恢复步数和评价脚本不再调整后，才会保存预注册式协议并对这套图进行一次性测试。Imagenette 源于 ImageNet，而公开 VQGAN 也使用 ImageNet 预训练，因此结果反映的是该受控的 ImageNet 派生条件，不能外推为完全未见视觉分布上的通信性能。

## 可复核文件

- `prepare_stage7_imagenette.py`：第三套数据划分、去重与来源哈希。
- `data/imagenette_stage7/manifest.json`：样本数、排除记录和测试锁定规则。
- `context_maskgit_v2.py`：二维局部—全局接收端。
- `train_context_maskgit_v2.py`：只读 train/val 的训练与消融脚本。
- `{args.run}/config.json`、`before.json`、`results.json`、`history.jsonl`：可复核训练证据。
'''
    (ROOT / 'STAGE7_REPORT.md').write_text(report, encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps({'best_step': results['best_step'], 'validation': aval, 'test_accessed': status['test_accessed']}, indent=2), encoding='utf-8')
    print(json.dumps({'best_step': results['best_step'], 'validation_correct': aval['correct']['accuracy'], 'test_accessed': status['test_accessed']}))


if __name__ == '__main__':
    main()
