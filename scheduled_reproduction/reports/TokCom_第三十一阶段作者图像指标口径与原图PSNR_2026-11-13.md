# TokCom 复现：第三十一阶段作者图像指标口径与原图 PSNR（2026-11-13）

## 本轮实测

固定第 26–30 阶段同样的十张开发图、AWGN 8 dB 第 0 轮物理接收、作者 VQGAN/MaskGIT 权重、真实类别和生成种子 `20261112`，重新运行作者 `GenDecoding`。原图按演示笔记本的 `Resize((256,256)) + ToTensor()` 处理，并逐图用文件 SHA-256 对照第 26 阶段缓存。按作者 `calculate_psnr` 的公式，在未裁剪的 VQGAN 输出和 `[0,1]` 原图之间计算 PSNR。

| 指标，十图平均 | 发送端真值比较遮蔽 | 接收端 CRC 失败遮蔽 |
|---|---:|---:|
| **对原图 PSNR，作者原始输出口径** | **19.06 dB** | **18.61 dB** |
| 对原图 PSNR，先裁剪到 `[0,1]` 的诊断值 | 19.22 dB | 18.83 dB |

两种 PSNR 不能混用。原始 VQGAN 解码输出可低于 0 或高于 1；例如 tench 图像第 0 轮的两分支输出分别有约 4.71% 和 5.79% 数值在 `[0,1]` 外。裁剪会改变 PSNR。十图中 garbage truck 的原始 PSNR 是真值比较 19.29 dB、CRC 19.37 dB，而其正确 token 数仍是真值比较 248/256、CRC 239/256，说明 token 正确数与原图 PSNR 的排序可以不同。

每张图重跑的 token 正确数和相对干净 VQGAN 重建的 MSE 均与第 28 阶段结果一致。逐图原始/裁剪后 PSNR 和输出范围比例见 `runs/stage31_author_psnr_summary.json`，十份逐图 JSON 见 `runs/stage31_author_metric_smoke_trial0_image*.json`。

## 作者评估代码核对

作者的[演示笔记本](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/TokCom_Demo.ipynb)把原图分别与生成图、直接接收解码图、干净 VQGAN 重建图比较；[指标工具](https://github.com/liqiao19/TokenCom_Code/blob/0422ed32dcde3347e0bb6697f343f7a9ff7b3214/Performance_metrics.py)中的 PSNR 未先裁剪生成图，LPIPS 函数却先把两张输入图裁剪到 `[0,1]`。因此第 28–30 阶段“对干净 VQGAN 图的 MSE”与本阶段“对原图的 PSNR”参考对象不同，不能直接相互换算。

LPIPS 上游实现默认期望 `[-1,1]` 输入；若使用 `[0,1]` 输入，应指定 `normalize=True`。作者工具将输入裁剪到 `[0,1]` 后直接调用模型，没有指定该选项。这意味着严格照作者工具复现的数值与常规规范化输入的 LPIPS 数值应分开标注，当前不能声称已经得到论文 LPIPS。[LPIPS 上游说明](https://github.com/richzhang/PerceptualSimilarity)、[上游 forward 实现](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py)。CLIP 仍未运行；作者代码将图像张量转成 PIL 后使用 ViT-B/32 图像编码器，输出范围和转换效果还需单独核验。

本轮只有十张开发图的一个物理接收轮次。两种遮蔽方式已知信息不同，不能由 0.45 dB 的样本均值差断言同等条件下的算法优势，也不能外推为论文总体性能。下一步应先固定 LPIPS/CLIP 的输入转换及版本，再扩展图像指标到五轮接收和新的留出图。
