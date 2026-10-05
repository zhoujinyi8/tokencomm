# 第五阶段：扩大数据训练与新留出集建设

本阶段建立了与旧pilot完全不重叠的新Imagenette子集：每类80张训练、10张验证、10张留出测试，共800/100/100张。旧pilot的240张训练图和80张验证/测试图分别从对应源划分排除；新划分同时检查文件SHA256与解码像素摘要，未发现跨划分重复。新的100张测试图尚未被训练、缓存或模型选择脚本读取。

## 1. 数据身份与边界

- 数据随机种子：20260928。
- 训练/验证/测试：800/100/100。
- 旧pilot重叠：0。
- VQ token缓存仅包含：train, val；`test_opened=false`。
- ResNet特征缓存调用同样只指定train和val；训练配置均记录`test_accessed=false`。

数据源仍为Imagenette。ResNet18在ImageNet上预训练，因此TaiChi实验不满足预训练阶段未见数据的严格条件；本阶段的“不重叠”指与本项目旧pilot和当前下游训练/验证划分不重叠。

## 2. 预训练视觉TaiChi

模型从第三阶段第500步最佳检查点热启动，在800张新训练图上更新1600步。每步按类别均衡抽样20张；前200步在AWGN 120 dB下训练适配器、BAN和KAN，之后联合训练小型语言/信道模块并在0–15 dB采样SNR。验证模型选择仍以AWGN 10 dB完整回答准确率优先、回答CE次优。

最佳检查点为第800步。冻结低分辨率特征的无通信线性分类基线在100张验证图上为91.0%。通信问答模型结果：

| 信道 | SNR / dB | 正确特征 | 错配特征 | 零特征 |
|---|---|---|---|---|
| AWGN | 10 | 84.0% | 0.0% | 10.0% |
| AWGN | 120 | 85.0% | 0.0% | 10.0% |
| Rayleigh | 10 | 75.0% | 2.0% | 11.0% |


AWGN 10 dB下达到84%，比新验证集的10%类别基线高，且错配特征降至0%；Rayleigh 10 dB两个种子为76%和74%。结果表明扩大训练数据后，视觉特征经过BAN/KAN、信道压缩和文本解码仍能保留较强类别信息。但这些是用于模型选择和诊断的验证结果，尚不是新留出集成绩。

## 3. 无CMI MaskGIT

模型使用800张训练图、100张验证图，始终输入无类别条件；公开VQGAN冻结。训练2400步，batch8，PER=30%、每包4 token，验证每400步一次。验证CE持续下降，因此选择第2400步。

| 验证恢复条件 | 丢失token精确恢复率 |
|---|---|
| 正确上下文 | 1.185% |
| 错配上下文 | 0.268% |
| 无上下文 | 0.204% |
| 空间填充 | 2.192% |


正确上下文明显好于错配和完全移除上下文，说明模型在新验证图片上开始利用已收到token，而不只是类别模板记忆。不过1.185%仍低于空间填充2.192%，未形成优于简单基线的恢复器。训练图正确上下文为1.902%，也仅略低于对应空间填充，扩大数据抑制了第二阶段那种小样本记忆，但模型容量、训练目标或预训练程度仍不足。

## 4. 结果图

![第五阶段训练曲线](runs/stage5_report/training.png)

![第五阶段验证集对照](runs/stage5_report/validation_controls.png)

## 5. 文件与复跑

- `data/imagenette_stage5/`：新划分、清单、训练/验证token及视觉特征缓存。
- `runs/stage5_pretrained_taichi/`：TaiChi最佳/最后权重、线性基线和验证消融。
- `runs/stage5_context_maskgit/`：MaskGIT最佳/最后权重、训练日志与上下文对照。
- `prepare_stage5_imagenette.py`：排除旧pilot并建立新划分。

```powershell
python prepare_stage5_imagenette.py --train-parquet ../../work/downloads/imagenette_train.parquet --val-parquet ../../work/downloads/imagenette_validation.parquet --out data/imagenette_stage5
python cache_pretrained_vision.py --data data/imagenette_stage5 --splits train val
python cache_real_tokens.py --data data/imagenette_stage5 --splits train val
python train_pretrained_taichi.py --data data/imagenette_stage5/resnet18_features --init runs/stage3_pretrained_taichi_v2/best.pt --steps 1600 --out runs/stage5_pretrained_taichi
python train_context_maskgit.py --data data/imagenette_stage5/token_cache --per-class 80 --steps 2400 --eval-every 400 --out runs/stage5_context_maskgit
```

## 6. 下一步

当前配置应先冻结，再首次缓存和评估新的100张测试图。测试协议需要在打开测试前记录信道条件、随机种子、检查点SHA256和评价指标，测试后不再调参。若后续继续开发模型，应另建新的留出集。
