# 第八阶段：ViT 与 ConvNeXt 双独立视觉主干 TaiChi

## 阶段目标

此前 TaiChi 使用同一个 ResNet18 的双分辨率特征。本阶段改为两个结构独立的公开预训练视觉主干：ViT-B/16 提供 Transformer patch token，ConvNeXt-Tiny 提供卷积特征；两路经过独立适配器后进入 BAN、KAN、语义信道和回答解码器。这使视觉侧更接近论文的双主干设计。报告只使用 Stage-7 的 1,000 张训练图和 150 张验证图，新的 150 张测试图保持未读。

## 权重与特征身份

- ViT-B/16 IMAGENET1K_V1 SHA256：`c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`。
- ConvNeXt-Tiny IMAGENET1K_V1 SHA256：`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`。
- 数据清单 SHA256：`56fc55312e3a5262cb7af427452146f002295cd7b0de8c9e60e31e8006935b87`。
- 缓存包含 train/val 共 1,150 张，`test_opened=false`；训练状态 `test_accessed=false`。
- ViT 去除 CLS 后将 14×14 patch token 汇聚为 768×4×4；ConvNeXt 使用最终 768×7×7 特征。

## 训练配置

模型从 Stage-5 最佳 TaiChi 检查点继承语言、融合和通信权重；两个新适配器重新初始化。前 200 步在 AWGN 120 dB 下训练适配器、BAN 与 KAN，随后联合训练至 1200 步并在 0–15 dB 随机采样。每步按类别均衡抽取 20 张图。模型含 540,628 个可训练参数，按固定 AWGN 10 dB 验证完整回答准确率优先、交叉熵次优选择检查点，最佳步为 1200。

![训练曲线](runs/stage8_report/training.png)

## 验证集结果

| 信道 | SNR / dB | 正确特征 | 错配特征 | 零特征 |
|---|---|---|---|---|
| AWGN | 10 | 98.67% | 0.00% | 10.00% |
| AWGN | 120 | 98.67% | 0.00% | 10.00% |
| Rayleigh | 10 | 90.00% | 1.00% | 10.00% |


![验证消融](runs/stage8_report/validation_controls.png)

AWGN 10 dB 正确特征平均准确率为 98.67%，错配为 0.00%，零特征为 10.00%；Rayleigh 10 dB 正确特征平均为 90.00%。每个均值来自同一批 150 张验证图上的两个信道种子，不是独立测试，也不是 300 张不同图。

## 解释范围

双独立主干明显提高了当前受控任务的验证准确率，并通过错配/零特征对照证明回答依赖视觉表示。它仍使用 ImageNet 分类预训练而非论文的完整 CLIP 对齐权重，语言模块仍是小型回答解码器。Imagenette 来自 ImageNet，因此不能声称视觉主干在预训练阶段未见过该分布。Stage-7 测试集继续锁定，后续应先冻结 Stage-8 配置再一次性测试，或继续采用 Qwen-LoRA 大语言模型路线并另建开发/测试协议。
