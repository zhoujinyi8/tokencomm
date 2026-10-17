# TokCom 与 TaiChi：复现工程

**第十三阶段TokCom大数据训练正在运行：** 第一篇论文路线现使用公开MaskGIT VQGAN的1024码本、256个图像token，在8,000张训练图和150张固定开发图上训练无CMI的ContextMaskGIT v2。冻结预算为6,000步、batch 16，并在PER 10%/30%/50%间做混合课程。缓存采用45秒可恢复片段；训练在完成跨片段恢复验证后采用最长240秒片段，并保留每100步原子检查点、单写者后台调度和完整优化器/RNG恢复。每500步进行三种子验证，测试集保持未读。当前状态见[STAGE13_PROGRESS.md](STAGE13_PROGRESS.md)，冻结配置见[STAGE13_TOKCOM_PROTOCOL.json](STAGE13_TOKCOM_PROTOCOL.json)。

Stage-13训练完成后，`evaluate_stage13_physical.py`会自动冻结最佳检查点哈希，并在同一开发集上执行CRC-16、卷积码、16-QAM、软Viterbi的AWGN/Rayleigh物理链路扫描。冻结网格为0/5/10/20 dB、每条件三个种子，共24个条件；该扫描报告PER、BER、未检出错误和端到端token准确率。`report_fair_comparison.py`随后按共同的信道与Es/N0标签生成资源和鲁棒性并排报告，同时把两条路线不同的任务指标和不同证据等级明确分开。由于TokCom扫描使用模型选择开发集，它只作工程诊断，不作盲测结论。

Stage-14盲测流水线也已在任何新测试图被选择前写定，完整预注册口径见[STAGE14_HOLDOUT_PLAN.md](STAGE14_HOLDOUT_PLAN.md)。它必须等Stage-13训练、开发扫描和公平比较全部完成，之后单独冻结最佳检查点、实现哈希、数据抽样种子和24个最终条件；下一次调度才会从公开Imagenette验证源中抽取300张与全部历史数据（含Stage-12留出集）文件哈希和像素哈希零重叠的新图。测试结果会单独生成`STAGE14_TOKCOM_FINAL_REPORT.md`与审计文件，且不再反馈到训练或模型选择。

**第十二阶段新留出集最终测试已完成：** Stage-11胜出的BAN+KAN第800步模型在300张新独立图上完成24个AWGN/Rayleigh信道条件和3个视觉控制。AWGN 0–20 dB为98.22%–98.67%，Rayleigh 0 dB为89.78%，5–20 dB为98.22%–98.67%；正确、错配和全零视觉控制分别为98.67%、0%和10%。27/27条件、数据零交集和`used_for_training=false`均已审计。详见[STAGE12_FINAL_TEST_REPORT.md](STAGE12_FINAL_TEST_REPORT.md)与[STAGE12_FINAL_AUDIT.json](STAGE12_FINAL_AUDIT.json)。

**第十一阶段等预算融合消融已完成：** BAN+KAN、BAN+MLP、Concat+MLP均使用同一8,000张训练图、100张均衡开发图、800步和有效批量8。按冻结的最低验证损失规则，BAN+KAN以0.001940和100%胜出；BAN+MLP为0.007914/99%，Concat+MLP为0.010666/99%。三者错配视觉特征均为0%，零特征为8%–10%。恢复审计证明三组都完整覆盖1–800步，重算步损失逐值一致，Stage-10最终测试集未读取。详见[STAGE11_ABLATION_REPORT.md](STAGE11_ABLATION_REPORT.md)、[STAGE11_RECOVERY_AUDIT.json](STAGE11_RECOVERY_AUDIT.json)和[STAGE11_PROGRESS.md](STAGE11_PROGRESS.md)。

**统一通信开销已完成可执行核算：** [COMMUNICATION_BUDGET_REPORT.md](COMMUNICATION_BUDGET_REPORT.md)分别报告离散token源码率、CRC/FEC后的编码比特和复信道符号，并明确连续TaiChi语义向量在没有量化器时不能换算为bpp。公开VQGAN规格配合当前参考数字链路为2,560源比特、7,936编码比特和1,984个复符号；Stage-10/11 TaiChi为1,024个实信道值，即512个复符号。两条链路任务不同，该数值只用于工程核算，不作为性能优劣结论。

**第十阶段8,000张图强化训练及预注册最终测试已完成：** Qwen2.5-0.5B双视觉语义通信模型从第400步继续训练到第2,400步，有效批量8。开发验证在第2,200步达到100%完整回答准确率、0.000164损失。冻结协议后一次性开启150张独立测试图：AWGN四个SNR三种子均值为99.33%–99.56%，Rayleigh为99.78%–100%；最终消融为正确特征100%、错配特征0%、零特征10%。测试集现已用于最终评价，禁止再据此调参。详见[STAGE10_FINAL_TEST_REPORT.md](STAGE10_FINAL_TEST_REPORT.md)、[STAGE10_REPORT.md](STAGE10_REPORT.md)、[STAGE10_CHANNEL_REPORT.md](STAGE10_CHANNEL_REPORT.md)和[STAGE10_RESEARCH_LOG.md](STAGE10_RESEARCH_LOG.md)。

**第九阶段大模型训练已完成：** 新增约5亿参数的Qwen2.5-0.5B-Instruct语言主干，基础权重冻结，注意力层使用LoRA；ViT-B/16与ConvNeXt-Tiny视觉特征经BAN、KAN和语义信道形成16个视觉前缀token。模型共496,264,896个参数，其中2,232,128个可训练（0.45%）。修正缩减验证集的类别偏置后，从第100步检查点恢复并完成400步CPU训练；固定十类别均衡验证的完整回答准确率由第200步31.25%提高到第400步48.44%。第400步在独立信道种子的视觉消融中，正确特征56.25%、错配特征0%、零特征10.94%。检查点位于`runs/stage9_qwen_dual_taichi_balanced/`，Stage-7测试集仍未打开。详见[STAGE9_REPORT.md](STAGE9_REPORT.md)。

**第八阶段双独立视觉主干已完成：** 公开ImageNet预训练ViT-B/16与ConvNeXt-Tiny替换共享ResNet18双尺度特征，在1000/150训练验证图上完成1200步训练。150张验证图两个信道种子均值为AWGN10dB 98.67%、AWGN120dB 98.67%、Rayleigh10dB 90.00%；AWGN错配特征0%、零特征10%。缓存只含train/val，`test_opened=false`，训练记录`test_accessed=false`。详见[STAGE8_REPORT.html](STAGE8_REPORT.html)和[STAGE8_REPORT.md](STAGE8_REPORT.md)。

**第六阶段独立测试与下一版MaskGIT接收端已完成：** 扩大数据模型在新的100张留出图上，TaiChi为AWGN10dB 81.8%、Rayleigh10dB 71.8%；MaskGIT仍低于空间填充。测试协议与结果见 [STAGE6_REPORT.html](STAGE6_REPORT.html)。新增 `context_maskgit_v2.py` 提供二维行列位置编码、局部深度可分卷积和全局Transformer的无CMI接收端；尚未用已测试的数据调参。

**第五阶段扩大数据训练已完成：** 新建与旧pilot不重叠的800/100/100训练、验证、留出测试划分；新的测试集尚未缓存或读取。预训练视觉TaiChi验证集AWGN10dB达到84%，Rayleigh10dB为75%均值；无CMI MaskGIT验证恢复率提升至1.185%，但仍低于空间填充2.192%。详见 [STAGE5_REPORT.html](STAGE5_REPORT.html) 和 [STAGE5_REPORT.md](STAGE5_REPORT.md)。

**第四阶段独立测试已完成：** 第三阶段检查点冻结后，首次开启40张保留测试图；测试协议预先保存，测试后未训练或改选模型。预训练视觉TaiChi在5个新信道种子下，AWGN10dB平均准确率60%，Rayleigh10dB为57.5%；MaskGIT仍弱于空间填充。详见 [STAGE4_REPORT.html](STAGE4_REPORT.html) 和 [STAGE4_REPORT.md](STAGE4_REPORT.md)。该测试集现已用于评价，后续调参应另建最终留出集。

**第三阶段已完成：** MaskGIT每类4图、完全无类别标签训练1200步；预训练ResNet18双分辨率特征接入TaiChi并完成1200步训练。TaiChi最佳检查点在40张验证图、两个噪声种子下AWGN10dB平均准确率71.25%，错配特征为0%；这仍是验证结果，非最终测试或论文CLIP/LLM复现。报告：[STAGE3_REPORT.html](STAGE3_REPORT.html)、[STAGE3_REPORT.md](STAGE3_REPORT.md)。单图入口为infer_pretrained_taichi.py，新模型使用专用加载器，不能交给旧infer.py加载。

**第二阶段诊断已完成：** 新增验证集图像消融、视觉梯度检查，以及两模型各800步的10图拟合实验。TaiChi能记住10张图但泛化仍弱；MaskGIT即使全部token擦除也能凭正确类别重建这10张训练图，说明该拟合设置存在类别模板记忆。详见 [STAGE2_REPORT.md](STAGE2_REPORT.md)。新诊断权重与第一阶段权重分别保存。

**最新真实数据训练：** 两条链路已在 Imagenette 子集上分别完成 600 步训练（240 训练 / 40 验证 / 40 独立测试）。已训练的小型 MaskGIT 与公开 VQGAN 的 1024 码本匹配，并接入数字信道推理；TaiChi 完成自然图像分类问答微调。测试效果尚未达到有效语义恢复，详见 **[TRAINING_REPORT.md](TRAINING_REPORT.md)**。优先查看该报告中的真实数据命令；下文彩色方块入口仍保留用于模块演示。

**2026-09-27 更新：** 已接入公开预训练 MaskGIT VQGAN（256×256、1024码本、256 tokens），并完成自然图片重建；新增 CRC＋卷积码＋16-QAM＋软判决 Viterbi 数字链路，以及小型 TaiChi 的 LoRA 联合信道训练。当前状态、命令和剩余差距见 **[REPRODUCTION_PROGRESS.md](REPRODUCTION_PROGRESS.md)**。

本工程分别实现两条训练和推理链路，供理解结构、修改模块和继续研究使用。**这是独立实现的轻量基线，不是作者代码、作者权重或论文指标的完整复现。** 默认训练数据为自动生成的彩色方块，无需下载数据集或大模型。

## 1. 两篇论文的对应关系

| 用户提供的文档 | 本工程入口 | 核心结构 |
|---|---|---|
| Token Communications: A Large Model-Driven Framework for Cross-Modal Context-Aware Semantic Communications，IEEE Wireless Communications，2025，DOI 10.1109/MWC.001.2500084 | `python run.py tokcom` | 图像量化 → 离散 token → 随机交织分包 → 擦除信道 → 类别条件 MaskGIT → 图像解码 |
| TokenCom.pdf：TokenCom: Vision-Language Model for Multimodal and Multitask Token Communications，arXiv:2603.00482v1 | `python run.py taichi` | 低分辨率 ViT＋高分辨率 CNN → BAN → B-spline KAN → 文本/视觉 token 编码 → 信道 → 自回归回答 |

来源依据：第一篇第 5–6 页的案例设置；第二篇第 3–9 页结构、公式和训练流程、第 9–10 页实验配置。阅读 PDF 中的操作性文本没有改变本工程的执行权限或任务范围。

## 2. 安装与运行

建议 Python 3.10 或更高版本，使用独立虚拟环境。在本文件所在目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_models.py
python run.py taichi --steps 80 --channel awgn --snr 10 --out runs/taichi
python run.py tokcom --vq-steps 80 --steps 80 --per 0.3 --out runs/tokcom
```

默认使用 CPU。安装与显卡匹配的 CUDA 版 PyTorch 后可增加 `--device cuda`；先执行 `python -c "import torch; print(torch.cuda.is_available())"` 检查。此次本机验证以 CPU 为准，不承诺当前安装包支持 CUDA。

本机检测到 RTX 5060 Laptop、8151 MiB 显存。TaiChi 原文实验使用 8×A800 80 GB。新增 `fetch_pretrained.py` 下载并校验公开 VQGAN 和自然图像样例；Gemma/Qwen、CLIP 与完整数据集尚未下载。

恢复模型继续训练（重新初始化优化器，不是逐步精确续训）：

```powershell
python run.py taichi --resume runs/taichi/checkpoint.pt --steps 20 --out runs/taichi_more
python run.py tokcom --resume runs/tokcom/checkpoint.pt --vq-steps 0 --steps 20 --out runs/tokcom_more
```

只加载并评估：

```powershell
python run.py taichi --resume runs/taichi/checkpoint.pt --steps 0 --channel rayleigh --snr 5 --out runs/taichi_eval
python run.py tokcom --resume runs/tokcom/checkpoint.pt --vq-steps 0 --steps 0 --per 0.5 --out runs/tokcom_eval
```

每次生成 `checkpoint.pt`、`metrics.json`；TokCom 额外生成 `reconstruction.png`，从左到右是原图、无丢包量化重建、丢包后恢复重建。指标只针对当前测试数据；合成数据结果不能当作论文的 ImageNet100/VQA 结果。

### 加载现成检查点进行单图推理和通信条件扫描

交付目录 `runs/taichi` 与 `runs/tokcom` 已包含本机 CPU 训练的初始演示检查点。**优先使用 `runs/tokcom_refined/checkpoint.pt`**：这是继续训练量化器和 MaskGIT 后的版本，重建质量明显改善。它们只学习过彩色方块，不能用于通用图片问答或高质量自然图像重建。完整验证记录见 `VALIDATION.md`。

```powershell
python evaluate.py --tokcom runs/tokcom_refined/checkpoint.pt --out runs/evaluation_refined --trials 3
python infer.py --checkpoint runs/taichi/checkpoint.pt --image runs/evaluation/sample_red.png --question "What color is the square?" --out runs/infer_taichi
python infer.py --checkpoint runs/tokcom_refined/checkpoint.pt --image runs/evaluation/sample_red.png --label 0 --per 0.3 --out runs/infer_tokcom
```

`evaluate.py` 在新种子的18张图上重复抽样信道，分别扫描 AWGN/Rayleigh 的 -5、0、5、10、20 dB，以及 PER=0、0.1、0.3、0.5、1；输出 `runs/evaluation/sweep.json` 和 `sweep.csv`。对有/无 CMI 使用相同擦除掩码，避免信道差异干扰比较。CSV 是不同条件下的实测小样本结果，不包含统计置信区间。

`infer.py` 可接收自己的图片，输出 `result.json`。TokCom 不传 `--label` 表示无类别条件；演示类别0/1/2分别是红/绿/蓝。自己的图片会被缩为32×32，输出质量受演示权重的训练范围限制。

继续训练版的生成命令（从初始检查点出发）：

```powershell
python run.py tokcom --resume runs/tokcom/checkpoint.pt --vq-steps 320 --steps 320 --per 0.3 --out runs/tokcom_refined
```

## 3. 第一篇：离散 token TokCom

```text
32×32 RGB → VQ encoder → 8×8 token IDs（码本64）
          → 随机交织，每包4个token → 按PER整包擦除
          → [MASK] + 可靠类别标签 → 双向Transformer迭代恢复
          → VQ decoder → 32×32 RGB
```

`VQTokenizer` 使用 straight-through 梯度、码本损失和 commitment loss；先训练量化器，再冻结它训练 `MaskGIT`。恢复时按余弦计划逐步填入最高置信度 token，正确收到的位置始终保留。训练中随机去掉 15% 类别条件，用同一个模型评估有/无 CMI。类别标签视为接收端可靠已知，符合论文案例假设。

明确差异：

- 原文使用预训练 VQGAN 和 MaskGIT，图像 256×256、256 个 token、码本 1024。`run.py tokcom` 保留 VQ-VAE 风格缩小后端；新增 `run_pretrained.py` 已接入匹配上述规格的公开预训练 VQGAN，但尚未接入匹配的预训练 MaskGIT Transformer。
- `run.py` 默认仍直接输入 PER。新增 `digital_link.py` 与 `evaluate_physical.py` 已实现 CRC、1/2 卷积码、16-QAM、软输入 Viterbi，测量 SNR→PER；新增参数中论文未披露的部分采用明确的参考配置，不能宣称与作者物理层逐项一致。
- 原文 256×10/256²≈0.0391 bpp 是源 token 开销。本工程 64×6/32²=0.375 bpp；均不能直接等同于包含 FEC/CRC/头部/侧信息的总传输开销。
- 轻量 `run.py` 入口仍只报告基础重建指标；Stage-16/17研究流水线已实现有限轮完整ARQ、语义置信度选择性ARQ、论文TCE公式，以及公开VQGAN解码后的PSNR、CLIP图像余弦和LPIPS v0.1评测。它们使用本工程参考物理层和开发集，不能解释为作者内部链路的精确复原。

接入真实 VQGAN 时应替换 `VQTokenizer.encode(image)->[B,N]` 和 `decode(ids)->image`，同时用匹配的码本大小与 token 数重建 `MaskGIT` 并训练或加载对应权重。现有 64 类 token 的权重不能直接用于 1024 码本。

## 4. 第二篇：TaiChi/BAN/KAN

默认张量路径：

| 模块 | 输入/输出 |
|---|---|
| LR ViT | 图像缩至 16×16，patch=4，输出 `[B,64,4,4]` |
| HR CNN | 32×32 图像，经两层步长2卷积，输出 `[B,64,8,8]` |
| BAN | 双向局部注意力、池化和残差，输出16个64维视觉token |
| KAN | LayerNorm＋两层 SiLU/B-spline edge functions，输出 `[B,16,64]` |
| Multimodal encoder | 拼接问题的 UTF-8 byte embeddings，2层Transformer |
| Channel encoder | 每token映射至16个实数，即8个复符号 |
| Channel | 单位平均复符号功率，AWGN或独立Rayleigh衰落 |
| Channel decoder | 还原64维表示 |
| Autoregressive decoder | 2层Transformer decoder，输出260词表的字节token |

关键实现解释：

- BAN 原文存在 high/low-level 与 HR/LR 命名混用，以及单位置 softmax 如何形成有效注意力的欠明确之处。这里明确以 `lr` 表示全局流、`hr` 表示细节流，按式(7)做 `lr + lr_reads_hr + pool(hr_reads_lr)`；每个 query 访问空间对齐后的 3×3 邻域。插值、窗口大小和单头注意力是本工程的选择，不宣称与作者内部实现一致。
- KAN 实现可训练的逐边三次 B-spline 系数与 SiLU 分支；固定均匀结点，不做自适应网格更新。`tanh` 将输入约束在样条定义域内。
- 信道 SNR 定义为 Es/N0，实部和虚部噪声方差各 N0/2。Rayleigh 使用接收端完美 CSI 和 MMSE 均衡；该细节为明确的工程假设。
- 使用因果掩码、右移答案，交叉熵只计算回答且忽略 PAD。答案不进入发送端编码器；接收端训练时只得到历史答案 token。
- 原文式(16)写法接近二元交叉熵且符号不统一，文本生成这里使用标准词表 categorical cross-entropy。
- 原文使用 CLIP-ViT-L/14-336、CLIP-ConvNeXt large d320 和 Gemma-2B/Qwen2.5-14B。这里分别用随机初始化小 ViT/CNN 和 encoder-decoder Transformer 替代；UTF-8 字节词表也不是原文 LLM tokenizer。
- 未实现真实 LLM 分层拆分、IDR/扩散生成、音视频输出与 DeepSpeed 多卡训练。新增 `train_lora.py` 在小型语言模块上实现了 LoRA 与 BAN/KAN/信道联合训练；尚未扩展到 Gemma/Qwen。当前模型不是具备通用知识的预训练 VLM。

`set_stage` 支持 `scratch/align/sft/joint`。默认 `scratch` 对所有模块从零训练，使合成数据闭环可验证。`align` 只解冻 BAN/KAN；`sft` 再解冻语言模块；`joint` 加入信道模块并全参数训练小语言模型。参数高效联合训练改用 `train_lora.py`；`infer.py/evaluate.py` 可自动加载 LoRA 检查点。非 scratch 阶段必须加载检查点；align/sft 需要 `--channel none`。

接入原文大模型时，需要先将两个视觉后端统一到 BAN 的通道维度，再把 KAN 输出维度改为 LLM embedding size，并重写 `encode/decode` 以适配特定 LLM 的位置编码、因果掩码、KV cache 与拆分位置。本工程提供模块边界，不提供已经验证的大模型适配器。

## 5. 用自己的数据训练

提供训练与独立验证 JSONL，每行一个样本，图片路径相对于 JSONL 目录：

```json
{"image":"images/0001.png","question":"What color is the car?","answer":"red","label":0}
```

```powershell
python run.py taichi --data data/train.jsonl --eval-data data/val.jsonl --steps 1000 --out runs/custom_taichi
python run.py tokcom --data data/train.jsonl --eval-data data/val.jsonl --vq-steps 1000 --steps 1000 --out runs/custom_tokcom
```

当前统一数据读取器要求四个字段；TaiChi 不使用 label，TokCom 不使用 question/answer。类别编号从0开始，验证集不能含训练集之外的类别。图片统一缩为32×32，每个文本截断至120个 UTF-8 字节；这是小模型验证限制，不适用于直接复现高分辨率细节或长文本任务。

## 6. 测试、可复现性与继续扩展

`test_models.py` 检查 B-spline 分割统一性及梯度、信道噪声功率随 SNR 的变化、端到端梯度、未来答案遮蔽、padding 隔离、丢包边界、已知 token 保留、VQ 码本梯度和冻结策略。程序固定随机种子，但 CPU/GPU/库版本不同仍可能导致数值差异。

训练输入仅使用训练集；默认验证方块采用独立种子。报告中保留环境、运行参数和逐步 loss。`--resume` 只加载模型参数，优化器状态不保存；继续训练 VQ 后必须重新训练 MaskGIT 以适应码本变化。

建议扩展顺序：先确认这两条轻量链路 → 换成真实数据做过拟合检查 → 接入匹配的预训练模型 → 校准物理层与通信开销 → 最后复现论文评测。完整复现还需要作者未明确给出的超参数、数据划分、训练细节与适配代码，不能仅凭这两份 PDF 保证数值一致。

PyTorch 的 padding mask 约定参考官方说明：<https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html>。本工程对 padding 使用 True 表示忽略，并通过测试检查 padding 不影响解码结果。

## 7. Stage-11/12 大规模融合消融与新留出集结果

在 Qwen2.5-0.5B 冻结主干、rank-8 LoRA、8,000 张训练图、800 个优化步和相同语义信道预算下，Stage-11 比较了 BAN+KAN、BAN+MLP 与 Concat+MLP。三者开发集最低损失分别为 0.001940、0.007914、0.010666，完整回答准确率分别为 100%、99%、99%。按测试前冻结的“最低损失优先、准确率次之”规则，BAN+KAN 第 800 步检查点胜出。

Stage-12 在打开数据前冻结模型、检查点哈希、300 张新留出图的选择规则、24 组信道条件和 3 组视觉控制。27/27 条件均已完成；新留出集没有用于训练或模型选择。主要结果如下：

| 信道 | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---:|---:|---:|---:|
| AWGN | 98.22% ± 0.16% | 98.44% ± 0.16% | 98.56% ± 0.16% | 98.67% ± 0.00% |
| Rayleigh | 89.78% ± 0.31% | 98.22% ± 0.31% | 98.33% ± 0.00% | 98.67% ± 0.00% |

在 AWGN 10 dB、固定种子 5299 下，正确视觉、跨类别错配视觉和全零视觉的准确率分别为 98.67%、0% 和 10%。完整协议、指标、图表和适用边界见 `STAGE12_FINAL_TEST_PROTOCOL.json` 与 `STAGE12_FINAL_TEST_REPORT.md`。测试集已经消费，后续训练不得据此调参，并必须创建新的最终留出集。

## 8. Stage-15/16/17 上下文TokCom、选择性ARQ与感知指标

Stage-15在Stage-13无CMI接收端上增加可靠类别上下文、15% classifier-free context dropout和logit guidance，使用8,000张训练图继续训练2,000步。冻结开发集选择得到guidance scale 1.0；该类别CMI只产生较弱增益，因此仍归类为近似的上下文实现，不等同于论文使用大模型获得的跨模态CMI。

Stage-16在150张开发图、AWGN 5 dB和Rayleigh 10 dB、三个信道种子上比较无重传、三个置信度门限和最多四轮完整ARQ。AWGN下无重传、门限0.02和完整ARQ的token准确率/TCE分别为61.25%/25.60、78.71%/19.67和96.93%/15.63；Rayleigh对应为91.32%/25.60、94.95%/24.60和99.99%/23.31。30/30条件的CRC未检出错误均为0。

Stage-17按冻结规则选择门限0.02，在同一开发集上完成无CMI、CMI无重传、CMI选择性ARQ和完整ARQ的24个条件、3,600次图像评价。AWGN下选择性ARQ相较无重传把CLIP余弦从约0.72提高到0.81、LPIPS从约0.29降低到0.20，同时TCE由25.60降至19.67；完整ARQ达到约0.86/0.14和15.63。Rayleigh 10 dB下信道本身较可靠，四种模式差距较小。结果属于开发评价，不是新的盲测；协议、审计和机器摘要见`STAGE17_PERCEPTUAL_PROTOCOL.json`、`STAGE17_AUDIT.json`与`STAGE17_PERCEPTUAL_SUMMARY.json`。

## 9. Stage-18 强类别上下文接收端

Stage-18从Stage-15的V3检查点精确热启动，增加零初始化输出FiLM和十类独立的1024码先验，并以0.15权重加入类别先验辅助损失。零初始化保证训练开始前V4与V3输出逐元素一致；辅助损失属于本工程增强方案，并非论文披露的原始训练式。训练继续使用8,000张公开Imagenette训练图、150张冻结开发图、PER 10%/30%/50%课程和3,000步预算，Stage-14盲测集没有读取。

3000/3000步训练、12个预定验证点和恢复状态审计全部通过，最佳检查点为第3,000步。相同三种子、PER 30%下，V4正确类别上下文的平均CE为6.018172，低于其无上下文6.022902，也低于V3正确上下文基线6.063891；精确token准确率为3.0499%，V3为2.8645%。guidance scale 1.0取得最低正确上下文CE，错误类别在较大guidance下持续恶化，表明接收端开始使用类别信息。准确率绝对值仍低，Stage-18只证明开发集上的条件建模得到加强，不能宣称达到论文结果。机器摘要、审计与图表见`STAGE18_SUMMARY.json`、`STAGE18_AUDIT.json`和`figures/stage18_*.png`。

## 10. Stage-19 V4端到端信道结果

Stage-19在与Stage-17完全相同的150张开发图、AWGN 5 dB/Rayleigh 10 dB、三个种子和四种ARQ模式下替换V4接收端，完成24/24条件、3,600次逐图评价。审计重新计算全部聚合值，核对240张预览、通信准确率、VQGAN基准和组合哈希；测试集与Stage-14留出集均未访问。

V4在无CMI/无ARQ下相对V3的改进较小：AWGN的token准确率提高0.0556个百分点、PSNR提高0.156 dB、CLIP提高0.00408、LPIPS降低0.00401；Rayleigh对应提高0.0113个百分点、0.0483 dB、0.00148和降低0.00086。直接沿用V3选择出的置信度门限0.02则出现质量回退：AWGN token准确率下降1.7066个百分点，Rayleigh下降0.4210个百分点。原因是V4置信度分布改变，使每包平均传输次数分别从1.3016降至1.2782、从1.0407降至1.0356；TCE提高但重传保护减弱。该结果要求在V4开发集上重新冻结门限，而不能事后把V3门限视为通用。完整结果与审计见`STAGE19_V4_LINK_SUMMARY.json`和`STAGE19_AUDIT.json`。

## 11. Stage-20/21 101类规模迁移

ImageNet官方图像需要登录或申请访问，当前无人值守流程不绕过其条款。Stage-20因此使用可直接下载的Food-101做101类、7-bit可靠CMI规模迁移，并明确标为非ImageNet100近似。官方归档SHA-256为`d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4`；70,700张训练和5,050张开发记录已经按冻结哈希规则生成，25,250张官方测试图像像素保持在压缩包内且未用于训练或选择。

`context_maskgit_v5.py`新增固定CLIP类别文本向量和可训练投影，复制V4的图像token接收权重。新投影以V4无上下文向量为中心零初始化，因此训练前V5无上下文logits与V4逐元素一致；训练后才学习类别名称的跨模态语义。Food-101数据域与ImageNet不同，V5文本条件与论文内部大模型CMI也不同，后续结果只能作为101类规模与文本上下文消融。

Stage-21已用固定版本`openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3`和本地完整权重生成101×512单位归一化类别文本向量。提示模板为`a photo of {class}, a type of food`，权重、分词器文件、类别顺序、提示列表、序列化输出和原始张量均记录SHA-256；最大单位范数误差为1.79e-7。VQGAN缓存协议固定处理70,700张训练图和5,050张开发图，以32张为原子分片，可在中断后续跑；初始实测CPU吞吐约1.08张/秒。官方测试像素仍未解压或读取。

统一隐藏任务`TokenCom-Goal-5Hour-Advance`每5小时推进一次，每轮主动预算20分钟；VQGAN缓存最多运行17.5分钟，余下时间用于分片落盘、状态更新或训练区段评价，Windows任务上限为25分钟。旧Stage-19/20快照任务已停用。状态汇总写入本地`GOAL_5H_STATUS.json`，不会弹出PowerShell或命令提示框。

缓存完成后的Stage-22训练已经在观察完整缓存结果前预注册，计划SHA-256为`313a223084396f6ba01a7246b0912d60f3c80f24992c64e0196cc44a77bd8638`。训练固定使用70,700张训练图、12,000步、批量32、PER 10%/30%/50%课程，共384,000次样本抽取；每1,000步在每类10张、共1,010张冻结开发子集上评价正确类别文本、错误类别文本和无文本三种控制。模型按正确文本平均masked CE选择，平局才比较单步缺失token准确率。每个五小时触发的20分钟训练区段结束前另用每类1张做监控评价并保存模型、优化器、调度器和RNG状态；区段监控不参与最佳检查点选择。`STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json`必须等全部75,750张token完成并通过聚合审计后才能生成。

## 12. Stage-23 V5选择性ARQ预注册链路

Stage-23在Stage-22训练结果出现前固定阈值校准和链路评价方法。它沿用Stage-22的类别内SHA-256排序，但使用互不重叠的三个窗口：每类第0–9张共1,010张只用于Stage-22模型选择，第10–14张共505张只用于ARQ门限校准，第15–19张共505张只用于最终开发链路评价。三个窗口的真实Food-101索引已经通过唯一性和两两不相交测试；官方25,250张测试图仍保持封存。

校准在AWGN 5 dB和Rayleigh 10 dB下分别运行三个固定信道种子，扫描`0.0025/0.005/0.01/0.02/0.04/0.08`六个V5置信度门限，并包含不重传与四轮完整ARQ锚点。每个信道选择能够弥合“不重传→完整ARQ”token错误率差距至少75%的候选中平均TCE最高者；若没有候选达标，则按最低token错误率、最高TCE、最低门限依次决胜。评价子集不参与门限选择。随后只在独立505张开发图上比较无CMI、CMI无ARQ、CMI选择性ARQ和完整ARQ，共24个条件。`evaluate_stage23_v5_link.py`逐条件原子写入并可续跑，`finalize_stage23.py`重新计算选择、聚合值、子集隔离和全部哈希，再生成门限校准图与准确率/TCE权衡图。该阶段仍是开发集评价，不是官方测试结果。
