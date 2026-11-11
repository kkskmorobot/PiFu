# lg3 预注册协议：成功版 LGLA v1 门控平衡结构的 seed44 复现

`LG3_PROTOCOL_STATUS=PREREGISTERED`

`THIRD_LOCKED_SEED_IN_LGLA_REPLICATION_CHAIN=TRUE`

`STRUCTURE_LOCKED_TO_SUCCESSFUL_LGX_AND_LG2=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_LG3_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_LG3_DEVELOPMENT`

## 启动依据

成功修订版 lgx seed43 已通过官方 CSV 首轮筛选门及门控诊断门。其精确 seed42 复现 lg2 也完成
100 个 epoch；权威 best-Joint 同行（CSV 可见 epoch 74、bundle `epoch_0073`）相对 Plain Run18：
Binary IoU `-0.00039`、Image Macro-F1 `+0.01641`、Group Macro-F1 `+0.01150`、Joint
`+0.00622`，四项预注册门槛全部通过。lg2 local gate 均值 `0.4958`、总体标准差 `0.1964`，门控
诊断亦通过，没有重现 lg1 的坍缩。

独立 standalone/fused validation 重新前向的 Group Macro-F1 与 Joint 增量较小，未达到数值门；该
差异保留为复核限制。实验协议规定正式选择和配对判定使用训练期间保存到 checkpoint selection_state
的 alias 对应 CSV 同行，而 standalone 诊断只决定门控是否有限、非坍缩。为避免继续针对 seed42
validation 调结构，下一步不增加新模块，按原多种子路线完成最后一个 seed44。

## 冻结实验因素

相对 Plain Run24，唯一模型因素是用已锁定的 P3/P4 predicted-lesion global/local softmax-gate 分类头
替换 Plain P3/P4 global-average 分类头，其中包含权重固定为 `0.01` 的 classified-batch mean
gate-logit balance 正则。

lg3 相对 lgx/lg2 除 seed、实验标识和配对 Plain 基线外，不改变模型、损失、数据、增强、优化器、
学习率、早停、指标、选择、checkpoint 或诊断合同。不得新增 P5、GeM、ROIAlign、温度、LayerNorm、
门控硬裁剪、熵正则、类别权重、风险损失、label smoothing、采样器、新增强或新阈值。

## 冻结训练合同

- seed=`44`，fresh start=`models/yolo26m-sem.pt`，配对基线为 Run24；
- development train/val=`23,462/1,744`，测试集、外部集和 PAD-UFES-20 不进入开发；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- 分类 CE 无类别权重、无 label smoothing，image classification weight=`0.5`；
- gate balance=`(mean(g_local-g_global))^2`，weight=`0.01`，仅作用于官方分类标签样本；
- delayed utility-gated early stop、validation、Joint、风险约束、checkpoint v2 与 alias 均不变；
- 正式输出独占 `experiments/lg3_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed44`；
- 唯一入口默认 `resume=auto`，用户点击前正式结果目录必须不存在。

## 配对筛选与停止规则

只比较 lg3 与 Run24 各自 `best_joint_score` bundle 对应的同一 CSV 行，不拼接不同 epoch 的峰值。
Run24 锁定同行为：Binary IoU `0.76393`、Image Macro-F1 `0.54301`、Group Macro-F1
`0.60641`、Group Recall `0.62047`、MEL→NV `0.34694`、malignant→benign `0.25121`、
Joint `0.68063`。

通过须同时满足：Group Macro-F1 差值不低于 `+0.005`、Joint 不低于 `+0.003`、Image Macro-F1
不下降、Binary IoU 差值不低于 `-0.005`；冻结 validation 局部门均值须位于 `[0.05,0.95]`、
标准差不低于 `0.001` 且全部诊断有限。

lg3 完成后必须先停止并汇总三个 seed 的同 seed 配对差值、均值和样本标准差；不得自动进入 ab1、
ab2、sum2 或 test2，也不得根据 lg3 validation 再修改结构。

## 启动边界

唯一正式入口是 `tools/lg3/train_lg3_from_pycharm.py`。AI 可执行静态、数据边界、GPU、checkpoint
save/resume、prepare-only 与 Trainer 构造预检，但不得调用正式训练路径。长训练必须由用户亲自在
PyCharm 点击。
