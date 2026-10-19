# Stage-13历史暂停快照

> 本文件记录2026-10-08的暂停点，不代表当前运行状态。当前阶段见[README.md](README.md)；本地动态状态见`GOAL_5H_STATUS.json`。

- 暂停时间：2026-10-08 19:15（Asia/Shanghai）
- 一分钟编排任务：已禁用
- 四小时进度快照任务：已禁用
- 后台相关Python进程：0
- Git工作目录：暂停说明与最新Stage-13自动报告待归档

## 已安全完成

- Stage-13公开VQGAN token缓存：8,150/8,150；
- Stage-13 ContextMaskGIT v2训练：6,000/6,000步；
- 训练历史：1–6,000严格连续、无重复，12个500步验证点完整；
- 最佳检查点：第6,000步，SHA-256 `f8b8c0ab73177b048d6bb33522b257d07e8c6c48cecfc092d8a183e1152bd919`；开发PER=30%平均交叉熵6.120058、丢失token恢复率2.7285%；
- Stage-13最终PER扫描与四类上下文/空间控制：完成；
- `STAGE13_TOKCOM_REPORT.md`、`STAGE13_AUDIT.json`及三张图：完成；
- 最终测试集访问：否；用于训练：否。

## 暂停点

- Stage-13显式数字物理链路开发扫描：1/24；
- 已完成条件：AWGN、Es/N0=0 dB、seed=13201；
- 已冻结协议：`STAGE13_PHYSICAL_DEV_PROTOCOL.json`；
- Stage-14盲测协议：尚未冻结；
- Stage-14新留出图：尚未选择、物化或缓存。

## 恢复方法

在Windows PowerShell中执行：

```powershell
Enable-ScheduledTask -TaskName 'TokenCom-Stage13-Orchestrator'
Enable-ScheduledTask -TaskName 'TokenCom-Stage13-4Hour-Progress'
Start-ScheduledTask -TaskName 'TokenCom-Stage13-Orchestrator'
```

调度器会检测Stage-13训练已完成，从第2个物理链路条件继续。24个开发条件完成后，它会依次生成物理报告、公平比较，冻结Stage-14协议，然后才物化300张新留出图并执行盲测。不要手工删除`runs/stage13_physical_dev/results.json`或改动已冻结协议和检查点。

## 复现边界

Stage-13使用匹配公开MaskGIT规格的ImageNet VQGAN（256个token、码本1024），但ContextMaskGIT Transformer由本工程自行训练，不是TokCom作者或Google原始权重。当前缺少作者MaskGIT权重、跨模态CMI、重传/TCE、完整包头/同步/导频开销，以及CLIP/LPIPS评价；报告中的结果属于近似实现，不是论文数值的精确回放。
