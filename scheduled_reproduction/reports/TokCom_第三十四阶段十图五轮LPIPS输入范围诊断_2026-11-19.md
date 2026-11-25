# TokCom 第三十四阶段：十图五轮 LPIPS 输入范围诊断

完成十张开发图、AWGN 8 dB 五轮接收共 50 组的作者 GenDecoding 重建及 LPIPS 评估。沿用作者 VQGAN/MaskGIT 权重、真实类别和生成种子 20261112。每组两分支共享接收 token，分别用发送端真值比较和接收端 CRC 失败确定遮蔽位置。每张图先平均五轮，再对十张图求均值。

| LPIPS AlexNet，越低越好 | 真值比较遮蔽 | CRC 失败遮蔽 |
|---|---:|---:|
| 作者工具口径：两图裁剪到 [0,1]，默认 normalize=False | 0.170329 | 0.181279 |
| 归一化口径：两图裁剪到 [0,1]，normalize=True | 0.199062 | 0.211458 |
| 原图与干净 VQGAN 重建：裁剪后 normalize=True | 0.189839 | 0.189839 |

依赖为 lpips 0.1.4，AlexNet 主干使用 torchvision 的 ImageNet 预训练权重，LPIPS 线性层使用包内 v0.1/alex.pth。原图沿用演示的 Resize((256,256))+ToTensor，源图 SHA-256 与缓存逐图相符。作者口径对应固定作者仓库 Performance_metrics.py 中对两图裁剪后直接调用模型；归一化口径利用 LPIPS normalize=True 将 [0,1] 输入转换到 [-1,1]。两个口径的绝对数值不同，不能混用或直接称为论文指标。

所有 50 组重跑的 token 正确数与第 31–33 阶段相同，干净 VQGAN 参考 MSE 的差异均小于 1e-8。汇总见 runs/stage34_lpips_five_trials_summary.json，含十张图各自的五轮均值；逐组结果见 runs/stage34_lpips_trial*_image*.json。

这些仍是十张开发图的重复信道接收，图像统计单位为 10。真值比较使用发送端原 token，CRC 只使用接收端校验，两者信息条件不同。两分支与干净 VQGAN 参考之间的均值差只用于该开发集诊断；仍需新的留出图像、完整评估协议及 CLIP 等指标。

作者代码依据：固定提交 0422ed32dcde3347e0bb6697f343f7a9ff7b3214 的 Performance_metrics.py 与 TokCom_Demo.ipynb，位于 work/tokencom_official_reference。下一步固定 CLIP 图像转换流程和权重版本，再接入相同接收结果进行评估。
