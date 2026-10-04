# 真实图像训练记录（2026-09-27）

两条链路均完成 600 次优化器更新，并保存最佳模型、最后模型、优化器、调度器、随机数状态和逐步日志。当前是可运行的小规模实验，尚未复现论文性能；不能把 loss 下降等同于有效的语义通信。

## 数据与实验边界

使用公开 Imagenette 320px 数据镜像，10 类。原始训练集抽取每类 24 张，共 240 张；原始验证集分别抽取每类 4 张作为本实验验证集、每类 4 张作为测试集，分别 40 张。随机种子 20260927，文件和解码像素均检查跨划分重复。清单、来源、SHA256 与类别映射保存在 `data/imagenette_pilot/manifest.json`。模型训练和最佳检查点选择不使用测试集。

这是 Imagenette pilot，不能称作论文 ImageNet100 或 VQA 基准。公开 VQGAN 曾用 ImageNet 预训练，因此本实验测试集独立性只针对此次下游训练，不表示预训练阶段未见过相关图像。CMI 使用接收端可靠已知的真实类别，不包括类别预测或标签传输成本。

## 实际训练配置

| 项目 | TokCom 的 MaskGIT | 小型 TaiChi |
|---|---|---|
| 输入 | 冻结公开 VQGAN 的 256 个离散 token，码本 1024 | 32×32 自然图像与固定分类问题 |
| 可训练参数 | 1,090,560；4 层、128 维 Transformer | 515,476；小型 ViT/CNN、BAN、KAN、语言和信道模块 |
| 初始化 | MaskGIT 从随机参数开始 | 从先前合成数据模型开始，全参数微调 |
| 更新次数 / batch | 600 / 8 | 600 / 8 |
| 初始学习率 | 0.0003，余弦调度 | 0.0003，余弦调度 |
| 训练条件 | 随机余弦掩码、15% 类别条件丢弃 | AWGN，SNR 在 0–15 dB 采样 |
| 初始 → 最佳验证 CE | 7.7077 → 6.8534 | 9.8219 → 0.2219 |
| 最佳更新步 | 450 | 550 |

环境为 PyTorch 2.14.0+cpu，CPU 4 线程；未使用 CUDA。TokCom 的验证 CE 是被遮蔽 token 的交叉熵，TaiChi 是回答字节的 teacher-forcing 交叉熵，两者不可横向比较。TaiChi 本次任务只是用自由文本回答图像类别，并非完整视觉问答。

## 独立测试：保留负结果

40 张测试图像，每包 4 个 token，固定随机交织和整包擦除；各恢复方法共享相同掩码。下表只计算丢失位置的精确 token 恢复准确率，不用已收到 token 抬高成绩。

| 设定 PER | 空间邻近填充 | 训练集最高频 token | MaskGIT + CMI | MaskGIT 无 CMI |
|---|---:|---:|---:|---:|
| 10% | 2.059% | 0.392% | 0.294% | 0.392% |
| 30% | 2.258% | 0.484% | 0.516% | 0.484% |
| 50% | 2.152% | 0.596% | 0.519% | 0.557% |

PER=30% 时单次预测测试 CE 为 6.8561，训练集频率分布基线为 6.8623，改善很小。学习后的精确恢复能力仍弱于空间基线。

每类第一张测试图片，共 10 张，在 PER=30% 下平均 PSNR：无信道损伤的 VQGAN 重建 17.103 dB；空间填充 14.717 dB；MaskGIT+CMI 13.646 dB；无 CMI 15.218 dB。无 CMI 在这 10 张图上的 PSNR 较高，但不能据此推断普遍优越性。类别条件目前没有带来稳定收益。

TaiChi 在 40 张测试图上自由生成完整回答：AWGN 10 dB 为 5/40（12.5%）；Rayleigh 0 dB 为 4/40（10%）；Rayleigh 10 dB 为 5/40（12.5%）。均衡十分类的固定类别基线为 10%。低回答字节 CE 尚未转化为可靠的图像分类能力。

此外已跑通单张自然图像的完整数字链路：VQGAN → 交织分包 → CRC → 卷积码 → 16-QAM → Rayleigh → 软 Viterbi → CRC 擦除 → 已训练 MaskGIT → VQGAN 解码。8 dB 下 64 包中 27 包失败，实测 PER=42.1875%，重建 PSNR=11.192 dB。该单图结果验证可执行性，不代表平均性能。

## 复跑与继续训练

在项目目录执行。交付包包含 320 张已选图片和 token 缓存；缓存后的 MaskGIT 训练不需要重复下载 VQGAN。图像重建需要 `fetch_pretrained.py` 下载的约 218 MB VQGAN，压缩包不重复附带该权重。

```powershell
python -m pip install -r requirements-training.txt
python train_real_maskgit.py --steps 600 --out runs/maskgit_repeat
python train_real_taichi.py --steps 600 --out runs/taichi_repeat
python fetch_pretrained.py
python evaluate_real.py --maskgit runs/maskgit_repeat/best.pt --taichi runs/taichi_repeat/best.pt --out runs/evaluation_repeat
```

从原始公开文件重新准备数据（已有清单时准备脚本拒绝覆盖；先选择新的数据目录）：

```powershell
python download_imagenette.py --out ../../work/downloads
python prepare_imagenette.py --train-parquet ../../work/downloads/imagenette_train.parquet --val-parquet ../../work/downloads/imagenette_validation.parquet
python cache_real_tokens.py
```

中断续训使用 `--resume runs/<原运行>/last.pt`，保持原输出目录和原定步数、batch、学习率、评估间隔、设备与线程设置。增加训练预算则使用新目录和新调度：

```powershell
python train_real_maskgit.py --init runs/maskgit_imagenette/best.pt --steps 2400 --out runs/maskgit_more
python train_real_taichi.py --base runs/taichi_imagenette/best.pt --steps 2400 --out runs/taichi_more
python run_pretrained.py --image data/imagenette_pilot/images/test/0/768ac080a91ac5ccfaba.jpg --maskgit runs/maskgit_imagenette/best.pt --label 0 --channel rayleigh --snr 8 --out runs/digital_repeat
```

上述延长训练命令仅供继续实验，未声称已执行。`--init` / `--base` 重置优化器；`--resume` 恢复原运行状态。真实 MaskGIT 的类别编号见清单，不能与旧彩色方块模型的 0/1/2 混用；省略 `--label` 为无 CMI。

## 文件与后续重点

本轮回归验证：`python -m unittest -v test_models test_digital_link test_lora test_receiver test_real_training`，15 项全部通过（78.268 秒）。包括模型梯度与遮蔽、CRC/卷积码/调制解调、LoRA 冻结与重载、接收端上下文以及优化器和随机数状态恢复的一步等价性。测试通过验证实现约束，不证明论文性能。

- `runs/maskgit_imagenette/`、`runs/taichi_imagenette/`：best.pt、last.pt、配置、训练日志和完成状态。
- `runs/real_evaluation/metrics.json`：完整测试指标和逐样本回答。
- `runs/real_evaluation/training_curves.png` / `.svg`：训练与验证曲线。
- `runs/real_evaluation/comparison.png`：前三类示例；从左到右为原图、无损信道 VQGAN、空间填充、MaskGIT+CMI；其余类别有单独图片。
- `runs/real_digital_inference/`：真实数字链路单图输出。

下一步应先扩大训练数据并验证小样本拟合能力，再接入论文规模的预训练视觉/语言主干。单纯在这 240 张图片上延长训练不能保证解决泛化问题。TokCom 尚缺作者匹配的预训练 MaskGIT、完整上下文/重传策略及论文基准；TaiChi 尚缺 CLIP 双主干、Gemma/Qwen、真实拆分与 IDR。当前结果不支持“完成两篇论文指标复现”的结论。
