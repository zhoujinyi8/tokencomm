# Stage-12 新最终留出集计划（尚未冻结、尚未打开）

## 目的

Stage-10的150张最终测试图已经完成一次性评价，不能再用于Stage-11结构选择或新模型的最终指标。Stage-12将在Stage-11三种融合结构全部完成并选定唯一模型后，从公开Imagenette 320px验证源中建立新的无重叠留出集。

## 当前封存状态

- `STAGE12_FINAL_TEST_PROTOCOL.json`目前不存在。
- 未下载或读取用于Stage-12选择的上游验证Parquet。
- 未创建`data/stage12_final_holdout`。
- 未缓存新留出集的ViT或ConvNeXt特征。
- `prepare_stage12_holdout.py`在协议不存在、状态不是`frozen`、Stage-11未全部完成、检查点哈希不匹配或源Parquet哈希不匹配时都会拒绝运行。
- `cache_stage12_final_test.py`和`evaluate_stage12_final.py`同样在冻结协议不存在时失败关闭；前者逐图复核文件/像素哈希后缓存特征，后者严格检查计划条件、已有结果和测试污染标志，并支持逐条件恢复。

## 冻结后流程

1. 完成BAN+KAN、BAN+MLP、Concat+MLP各800步，并生成`runs/stage11_report/final_summary.json`。
2. 严格按冻结规则选择一个结构和对应`best.pt`，记录检查点SHA-256。
3. 创建并先提交`STAGE12_FINAL_TEST_PROTOCOL.json`，固定上游文件SHA-256、所有历史排除清单的SHA-256、选择种子、每类样本数、信道条件、随机种子和评价指标。
4. 下载公开Imagenette 320px验证Parquet；已知历史来源SHA-256为`fa51906de9c2847cd058d9896d6b76e97a4499e4e07efca14fdffb3ae2661847`，运行时必须再次校验。
5. 排除pilot、Stage-5、Stage-7和Stage-10所有训练/验证/测试记录的文件SHA-256与解码像素SHA-256，再按冻结种子做类别均衡抽样。
6. 先写入新留出集身份清单并标记`model_evaluation_accessed=false`，随后单独缓存双视觉特征并执行一次预注册评价。
7. 评价完成后将该测试集标为已消费；任何后续架构或训练变化都不得根据其结果调参。

评价器与报告器已经在协议冻结前实现。预定指标包括AWGN/Rayleigh多SNR完整回答准确率均值、信道种子标准差、回答损失、逐类别准确率、错误图片索引以及正确/错配/零视觉特征控制。`test_stage12_final.py`验证条件计划无重复、完整结果汇总、部分结果不可标记完成，以及未知/重复条件失败关闭。

公开来源为[Hugging Face上的ShaomuTan/imagenette 320px配置](https://huggingface.co/datasets/ShaomuTan/imagenette/tree/main/320px)，上游数据源为fastai Imagenette。计划文件不构成冻结协议，具体测试规模和评价条件要等Stage-11结果完成后一次性固定。
