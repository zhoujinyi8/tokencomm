# 第七阶段：ContextMaskGIT v2 的独立开发集训练与验证

## 本阶段目的

第六阶段的 100 张最终测试图已完成一次性评价，因此不再可用于任何结构修改、训练或模型选择。本阶段为新版 ContextMaskGIT v2 新建第三套 Imagenette 划分，以检验二维位置、局部图像先验与全局 token 上下文是否能改善 VQ token 的丢失恢复。报告中的所有数值来自训练集或验证集，**不包含新的 150 张最终测试图**。

## 数据与隔离协议

Stage-7 从公开 Imagenette 原始 Parquet 文件构建，每类采样 100/15/15 张，合计训练 1,000 张、验证 150 张、测试 150 张。划分排除了 pilot 和 Stage-5 的已使用图片；下载文件哈希、源图片 SHA256 与解码 RGB 像素哈希均记录在 `data/imagenette_stage7/manifest.json`。跨 split 的文件和像素重复数均为零。

公开 VQGAN 的预处理为 `PIL ImageOps.fit(256, LANCZOS) → RGB float[0,1]`，checkpoint SHA256 为 `424e7f2b11d1d63691ae3fbab74cf6c417ecbe0d8f86187b71f3538c7b56f6d2`。训练缓存含 1,150 张 train/val 图，缓存元数据为 `test_opened: false`；测试目录下没有 token 缓存。本次模型选择只以验证掩码交叉熵决定。

## 模型与训练

ContextMaskGIT v2 不输入类别、问答答案或文本描述。输入是 16×16 的 1,024 类 VQ token 网格；已丢失 token 替换为 mask code。模型将独立的行和列嵌入相加，以局部深度可分离卷积块建模邻域，再以 Transformer 建模全局关联。接收端在 2000 步训练中随机产生 30% 的、每包 4 token 的丢包，并采用 128 维隐层、4 个 Transformer 层和 2 个局部块。恢复采用 8 次置信度逐步填补，所有未丢失 token 原样保留。

![训练曲线](runs/stage7_report/training.png)

最佳验证检查点为第 **2000** 步。`best.pt`、`last.pt`、完整逐步训练记录和模型配置均位于 `runs/stage7_context_maskgit_v2_b16_fixed/`。

## 验证集上下文对照

下表使用固定随机种子、同一批丢失位置和 PER=30% 的验证条件。正确上下文从同一幅图的接收 token 取得；错配上下文改用不同类别图像；无上下文将输入全部遮蔽；空间填补是非学习的邻近复制基线。精确恢复率仅在原始丢失位置统计。

| 数据 | 条件 | 精确恢复率 | 统计的丢失 token | 掩码交叉熵 |
|---|---|---|---|---|
| train | 正确上下文 | 1.039% | 76908 | 6.5380 |
| train | 错配上下文 | 0.329% | 76908 | 7.0242 |
| train | 无上下文 | 0.165% | 76908 | 7.0652 |
| train | 空间填补 | 2.153% | 76908 | — |
| val | 正确上下文 | 0.947% | 11720 | 6.6005 |
| val | 错配上下文 | 0.307% | 11720 | 7.0041 |
| val | 无上下文 | 0.247% | 11720 | 7.0826 |
| val | 空间填补 | 2.014% | 11720 | — |


![验证对照](runs/stage7_report/validation_controls.png)

验证集上正确上下文相对错配上下文的差值为 **0.640 个百分点**；相对空间填补的差值为 **-1.067 个百分点**。该差值用于检查模型是否利用同图的可接收 token，而不是只学习码本的静态频率。该观察仅适用于这个开发验证集，不能解释为最终留出测试结论。

## 结果的范围与下一步

本阶段已经完成独立开发数据上的结构训练和验证。150 张 Stage-7 测试图仍保持未读；只有在确认本结构、训练步数、恢复步数和评价脚本不再调整后，才会保存预注册式协议并对这套图进行一次性测试。Imagenette 源于 ImageNet，而公开 VQGAN 也使用 ImageNet 预训练，因此结果反映的是该受控的 ImageNet 派生条件，不能外推为完全未见视觉分布上的通信性能。

## 可复核文件

- `prepare_stage7_imagenette.py`：第三套数据划分、去重与来源哈希。
- `data/imagenette_stage7/manifest.json`：样本数、排除记录和测试锁定规则。
- `context_maskgit_v2.py`：二维局部—全局接收端。
- `train_context_maskgit_v2.py`：只读 train/val 的训练与消融脚本。
- `runs/stage7_context_maskgit_v2_b16_fixed/config.json`、`before.json`、`results.json`、`history.jsonl`：可复核训练证据。
