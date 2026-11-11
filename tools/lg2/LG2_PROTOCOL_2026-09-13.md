# lg2 预注册协议：成功版 LGLA v1 门控平衡结构的 seed42 复现

`LG2_PROTOCOL_STATUS=PREREGISTERED`

`POST_LGX_REPLICATION_CHAIN=TRUE`

`STRUCTURE_LOCKED_TO_SUCCESSFUL_LGX=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_LG2_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_LG2_DEVELOPMENT`

## 启动依据

lgx 已完成 100 个 epoch。其权威 best-Joint 同行（CSV 可见 epoch 99、bundle `epoch_0098`）相对
seed43 Plain Run22 的 Binary IoU、Image Macro-F1、Group Macro-F1 与 Joint 分别提高约
`+0.01486`、`+0.00714`、`+0.02564` 与 `+0.02083`，四项预注册筛选门全部通过。冻结 validation
诊断的 local gate 均值为 `0.5151`、总体标准差为 `0.1988`，诊断门亦通过，证明 lg1 的局部门坍缩
已被门控平衡修订解决。

因此按 `PROJECT_LGLA_EXTENSION_PLAN_2026-09-12.md` 的多种子路线，不再依据 validation 追加或
改造模块，而将成功的 lgx 结构完整锁定并在 seed42 复现。lg2 与相同 seed42 的 Plain Run18 配对。

## 冻结实验因素

相对 Plain Run18，唯一模型因素是用 lgx 已锁定的 P3/P4 predicted-lesion global/local softmax-gate
分类头替换 Plain P3/P4 global-average 分类头，其中包括权重固定为 `0.01` 的 classified-batch
mean gate-logit balance 正则。lg2 相对 lgx 除 seed、实验标识和配对基线外不改变模型、损失、数据、
增强、优化器、学习率、早停、指标、选择、checkpoint 或诊断合同。

不得新增 P5、GeM、ROIAlign、温度、LayerNorm、门控硬裁剪、熵正则、类别权重、风险损失、label
smoothing、采样器、新增强或新阈值。

## 冻结训练合同

- seed=`42`，fresh start=`models/yolo26m-sem.pt`，配对基线为 Run18；
- development train/val=`23,462/1,744`，测试集、外部集和 PAD-UFES-20 不进入开发；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- 分类 CE 无类别权重、无 label smoothing，image classification weight=`0.5`；
- gate balance=`(mean(g_local-g_global))^2`，weight=`0.01`，仅作用于有官方分类标签样本；
- delayed utility-gated early stop、validation、Joint、风险约束、checkpoint v2 与 alias 均不变；
- 正式输出独占 `experiments/lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42`；
- 唯一入口默认 `resume=auto`，用户点击前正式结果目录必须不存在。

## 配对筛选与停止规则

只比较 lg2 与 Run18 各自 `best_joint_score` bundle 对应的同一 CSV 行，不拼接独立峰值。Run18
锁定同行为：Binary IoU `0.76323`、Image Macro-F1 `0.54417`、Group Macro-F1 `0.61023`、
Group Recall `0.59915`、MEL→NV `0.38776`、malignant→benign `0.27536`、Joint `0.68246`。

通过须同时满足：Group Macro-F1 差值不低于 `+0.005`、Joint 不低于 `+0.003`、Image Macro-F1
不下降、Binary IoU 差值不低于 `-0.005`；冻结 validation 局部门均值须位于 `[0.05,0.95]`、
标准差不低于 `0.001` 且全部诊断有限。

无论通过、边界或失败，lg2 完成后都先停止并向用户报告完整差值；不得自动准备或训练 seed44，
也不得在 validation 上继续堆叠新模块。

## 启动边界

唯一正式入口是 `tools/lg2/train_lg2_from_pycharm.py`。AI 可执行静态、数据边界、GPU、checkpoint
save/resume、prepare-only 与 Trainer 构造预检，但不得调用正式训练路径。长训练必须由用户亲自在
PyCharm 点击。
