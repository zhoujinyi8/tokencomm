# 复现推进记录（2026-09-27）

最新训练结果见 [TRAINING_REPORT.md](TRAINING_REPORT.md)：已在公开 Imagenette 子集上完成两条链路各 600 步训练、独立测试及已训练 MaskGIT 的数字信道推理。下面保留基础模块接入阶段的记录。

当前已完成：公开预训练VQGAN接入、真实自然图片量化重建、数字物理链路、接收端无原始token泄漏的恢复接口，以及小型TaiChi的LoRA联合信道训练。**这仍不是两篇论文的全部模型和指标复现。**

## 1. 已实现模块

| 模块 | 状态 | 入口 |
|---|---|---|
| 预训练MaskGIT VQGAN | 已下载、SHA256校验、strict=True加载、自然图片验证 | `pretrained_vq.py`, `run_pretrained.py` |
| CRC＋卷积码＋16-QAM＋信道＋软输入Viterbi | 已实现和测试 | `digital_link.py` |
| SNR→BER/PER实测 | 已完成AWGN/Rayleigh扫描，每个设置256包 | `evaluate_physical.py` |
| 小模型MaskGIT＋数字物理层 | 已接入单图推理和扫描 | `infer.py --link digital` |
| LoRA＋BAN/KAN/信道联合训练 | 100步训练完成、权重保存、重新加载推理通过 | `train_lora.py`, `lora.py` |
| 测试 | 14项全部通过 | `python -m unittest -v test_models.py test_digital_link.py test_lora.py test_receiver.py` |

## 2. 直接运行

在本文件所在目录执行。当前本机已具备模型权重；工程ZIP不包含218 MB的大权重，解压到其他机器后先运行下载脚本。

```powershell
python fetch_pretrained.py
python fetch_pretrained.py --verify-only

# 公开自然图片：原图、量化重建、经数字链路后的重建
python run_pretrained.py --snr 8 --out runs/pretrained_vqgan
python run_pretrained.py --snr 4 --out runs/pretrained_vqgan_snr4

# 测量论文token位宽下的物理层；并验证小模型接收端恢复
python evaluate_physical.py --packets 256

# 小模型：由数字物理链路产生错误，CRC识别失败包后交给MaskGIT
python infer.py --checkpoint runs/tokcom_refined/checkpoint.pt --image runs/evaluation/sample_red.png --label 0 --link digital --channel rayleigh --snr 8 --out runs/infer_tokcom_digital

# TaiChi小模型：只训练LoRA和BAN/KAN/信道模块
python train_lora.py --steps 100
python infer.py --checkpoint runs/taichi_lora/checkpoint.pt --image runs/evaluation/sample_red.png --channel rayleigh --snr 0 --out runs/infer_taichi_lora
```

`run_pretrained.py` 接受 `--image`、`--snr`、`--channel` 和 `--device`。输入图像中心裁剪并缩至256×256，像素范围[0,1]。其丢失token当前使用**接收端空间邻近填充对照**，未接入预训练MaskGIT；输出JSON明确记录此限制。不能把这一路线的填充结果说成MaskGIT语义恢复。

## 3. 已接入的公开模型和资源

- 权重：`fun-research/TiTok`发布的 `maskgit-vqgan-imagenet-f16-256.bin`，约218 MB。
- 固定revision：`ab646ed225080a3acb7c78440a574d7f67f16fa7`。
- SHA256：`424e7f2b11d1d63691ae3fbab74cf6c417ecbe0d8f86187b71f3538c7b56f6d2`，与原始模型发布页一致；本机通过镜像下载后校验。
- 配置：1024个码本项、码本维度256、256×256图像、16×16离散token，每个token10bit。
- 模型是公开MaskGIT tokenizer的社区PyTorch转换，**不是TokenCom作者专门发布的模型权重**。
- 结构文件固定在上游commit `ca5c7c39cf55d49fb034d7348595c9c3abc2c4ed`；原代码、许可证和署名保留在 `third_party/`。
- 自然图片取自PyTorch Hub公开的`dog.jpg`，仅用于单图端到端检查，不作为训练集或论文评测数据集。

资源链接：

- https://huggingface.co/fun-research/TiTok/blob/ab646ed225080a3acb7c78440a574d7f67f16fa7/maskgit-vqgan-imagenet-f16-256.bin
- https://github.com/xyfJASON/maskgit-pytorch
- https://github.com/google-research/maskgit
- https://github.com/pytorch/hub/blob/master/images/dog.jpg

## 4. 真实自然图片结果

来自 `runs/pretrained_vqgan/metrics.json` 与 `runs/pretrained_vqgan_snr4/metrics.json`。只有一张图、一个随机种子，不代表数据集平均指标。

| 设置 | 无信道量化重建PSNR | 信道后重建PSNR | CRC失败包 |
|---|---:|---:|---:|
| AWGN，Es/N0=8 dB | 20.41 dB | 20.41 dB | 0/64 |
| AWGN，Es/N0=4 dB | 20.41 dB | 13.30 dB | 48/64 |

8 dB样例在原始解调码比特中有773个错误，经过卷积码Viterbi译码和CRC后有效载荷零错误。4 dB样例有75%的包未通过CRC，邻近填充产生可见失真；后续接收端需要对这些擦除位置执行语义恢复。

完整码率核算（单图）：

- 256 tokens × 10 bit = 2560源比特，源开销 **0.0390625 bpp**，与论文案例规格一致。
- 每包4 tokens，40有效载荷bit＋16 CRC bit＋6终止bit，1/2卷积编码后124bit。
- 共64包，7936编码bit，即 **0.12109375 coded bpp**。
- 16-QAM每符号4bit，共1984复符号，即 **0.0302734375 channel uses/pixel**。
- 此处不含类别侧信息、包头、交织种子、导频和重传；不能把源bpp等同于完整无线传输开销。

## 5. 数字物理层选择与结果

原论文给出了CRC、1/2卷积码、16-QAM、Soft Viterbi，但未提供足够的底层参数。这里采用以下**参考选择**：CRC-16/CCITT-FALSE（0x1021，初值0xFFFF）；约束长度7、(171,133)八进制卷积码；6个零比特终止；Gray 16-QAM（平均符号能量1）；max-log bit LLR；以零状态结束的软输入Viterbi。

SNR定义为**Es/N0**，不是Eb/N0。Rayleigh逐符号独立衰落，接收端完美CSI，直接用信道增益计算星座距离。CRC未检出的错误不会被真值“自动修复”，而是保留在接收token内并单独统计。

每项256个独立的40bit有效载荷包，种子固定，结果见 `runs/physical_sweep/sweep.csv`：

| Es/N0 | AWGN实测PER | Rayleigh实测PER |
|---|---:|---:|
| 0 dB | 99.61% | 100.00% |
| 4 dB | 67.58% | 94.14% |
| 8 dB | 0.39% | 36.33% |
| 12 dB | 0.00% | 3.52% |
| 16 dB | 0.00% | 0.00% |

CSV提供Wilson 95%区间。0/256失败包并不意味着真实PER为零：此时95%区间上限约1.48%。该扫描的参数选择不能替代作者未披露的物理链路，也不能据此宣称复现了论文曲线。

`evaluate_physical.py` 同时输出6bit小码本演示链路，字段以`toy_`区分。它与10bit论文位宽的包长不同，不能混用其PER。

软判决Viterbi参考：https://www.mathworks.com/help/comm/ug/estimate-ber-for-hard-and-soft-decision-viterbi-decoding.html

## 6. TaiChi的LoRA联合训练

在已训练的小型TaiChi上，冻结视觉编码器、文本embedding、原始Transformer投影和输出头；仅更新：

1. 语言encoder/decoder的LoRA低秩增量。
2. BAN与KAN。
3. 信道encoder/decoder。

LoRA覆盖注意力合并QKV投影、注意力输出投影和前馈层，rank=4、alpha=8，B零初始化保证初始函数与基础模型一致。使用PyTorch参数化机制，使直接使用投影权重的MultiheadAttention路径也能得到低秩增量。测试检查所有LoRA B矩阵获得非零有限梯度、冻结权重不变和保存/加载数值一致。

运行结果：100步Rayleigh训练，SNR每步均匀采样[-5,15] dB，AdamW初始lr=2e-4并余弦衰减。总参数534932，LoRA参数19456，含BAN/KAN/信道共161616个可训练参数。**冻结参数逐项比较保持不变。**

新种子18张方块验证图，在单次固定噪声采样下：-5 dB为66.67%；0、5、10、20 dB均为100%。这是小模型/小样本验证，不代表Gemma/Qwen或真实VQA效果，不作统计显著性结论。`runs/taichi_lora/metrics.json` 保存参数量、LoRA目标和loss；检查点可由`infer.py`与`evaluate.py`加载。

LoRA初始化参考：https://huggingface.co/docs/peft/v0.7.0/conceptual_guides/lora

## 7. 剩余复现差距

第一篇：还需与当前tokenizer匹配的预训练MaskGIT Transformer、ImageNet100明确类别/划分、可靠类别标签映射、论文上下文恢复策略、重传/TCE以及CLIP/LPIPS评测。已核实Google官方发布入口；本机请求原始Transformer检查点URL返回HTTP 403，本次没有冒用其他码本权重来代替。

第二篇：CLIP-ViT-L/14-336和CLIP-ConvNeXt large d320双视觉主干、Gemma-2B/Qwen2.5-14B及真实拆分、论文数据预训练/SFT、IDR生成和分布式训练仍未完成。当前LoRA实现验证的是训练方法和模块冻结，不是预训练VLM本体。

本机继续使用CPU版PyTorch；RTX 5060 Laptop尚未完成CUDA运行验证。最新阶段已增加 Imagenette 320 张自然图像，按 240/40/40 划分，并完成两模型各 600 步训练；见 [TRAINING_REPORT.md](TRAINING_REPORT.md)。已接入自己训练的匹配码本小型 MaskGIT，但仍不是作者预训练 Transformer。本文其余部分记录此前阶段；最新指标以训练报告为准。
