# PiFuV2 Run13 正式结果登记

日期：2026-08-09  
实验：`experiments/13_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELWeightedCE`

`RUN13_TRAINING_EXECUTION_GATE=PASS_NORMAL_EARLY_STOP`  
`RUN13_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN13_FEASIBILITY_GATE=FAIL_NO_FEASIBLE_CHECKPOINT`  
`RUN13_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 训练与存档

- `results.csv` 含连续可见 epoch 1~72 共 72 行，非有限值为 0；
- 完成标记 `last_completed_epoch=71`，对应可见 epoch 72；
- stopper `best_epoch=52`、`patience=20`、`possible_stop=true`，正常早停；
- 6 个保留 bundle `epoch_0020/0051/0055/0069/0070/0071` 全部通过大小、SHA-256、
  fingerprint 和完整状态验证；
- auto resume 指向 `epoch_0071`；无异常、无中断请求；
- 最新 bundle 包含 model、EMA、optimizer、scaler、scheduler、stopper、selection、
  Python/NumPy/Torch CPU/CUDA RNG 和 RandomSampler generator state。

## 2. val-only 结果

- 最佳 binary IoU/Dice：可见 epoch 21，0.77632/0.87408；
- 最佳 image Macro-F1：可见 epoch 62，0.53763；
- 最佳 group Macro-F1：可见 epoch 56，0.59183；
- 最佳 joint：可见 epoch 52，0.67586；
- 最佳 joint 状态 MEL->NV=0.32653；
- 全程最低 MEL->NV：可见 epoch 1，0.15306；
- 冻结上限 0.1077844311，72 个 epoch feasible 数量为 0；
- `best_feasible_joint` alias 不存在。

## 3. 2×2 结论与决策

Run13 相比 Run12 的整体分类和 joint 均下降，最低 MEL->NV 虽从 0.25510 改善到
0.15306，但没有达到 Run11 的 0.11224，更未越过约束。全 MEL 权重与 BLG-MDA
attention 出现不利交互。

不继续已正常早停的 Run13。下一实验 Run14 回到 Run12 的 unweighted CE，仅加入
train-prior-derived、target-conditioned MEL-vs-NV competitor logit margin，直接处理
失败边界而不重权 MEL 对其他六类。Run13 所有结果与 bundle 原样保留；正式模型锁
仍未建立，禁止 test/external。
