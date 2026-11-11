# PiFuV2 Run12 正式结果登记

日期：2026-08-08  
实验：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`

`RUN12_TRAINING_EXECUTION_GATE=PASS_NORMAL_EARLY_STOP`  
`RUN12_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN12_FEASIBILITY_GATE=FAIL_NO_FEASIBLE_CHECKPOINT`  
`RUN12_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 训练与存档完整性

- `results.csv` 含连续可见 epoch 1~91 共 91 行，非有限值为 0；
- 完成标记为 `TRAINING_COMPLETE_OR_NORMAL_EARLY_STOP`，内部
  `last_completed_epoch=90`，对应可见 epoch 91；
- stopper 的 `best_epoch=71`、`patience=20`、`possible_stop=true`，与内部 epoch
  90 正常早停一致；
- 无异常标记、无安全中断请求；
- 5 个保留 bundle `epoch_0020/0040/0088/0089/0090` 均通过文件大小、SHA-256、
  fingerprint 和完整状态验证；
- 最新 bundle 含 live FP32 model、EMA、optimizer、scaler、scheduler、stopper、
  selection、Python/NumPy/Torch CPU/CUDA RNG 与 RandomSampler generator state；
- auto resume 指向 `epoch_0090`，但完成标记禁止覆盖或继续已完成 Run12。

## 2. val-only 统一结果

- 最佳 binary IoU / Dice：可见 epoch 21，0.77914 / 0.87586；
- 最佳 image Macro-F1：可见 epoch 40，0.55403；
- 最佳 group Macro-F1 / joint：可见 epoch 41，0.59969 / 0.68258；
- 最佳 joint 状态的 MEL->NV / malignant->benign：0.45918 / 0.30918；
- 全程最低 MEL->NV：可见 epoch 4，0.25510；
- 冻结上限为 0.1077844311，91 个 epoch 的 feasible 数量为 0；
- `best_feasible_joint` alias 不存在。

## 3. 与 Run10 / Run11 的受控结论

相对 Run10，Run12 的最佳 image Macro-F1、group Macro-F1、joint 分别提高约
0.01110、0.00543、0.00126，最低 MEL->NV 从 0.32653 降到 0.25510。BLG-MDA
注意力对整体分类和联合指标产生小幅正增益，也改善风险方向，但不足以满足约束。

Run11 的 MEL 权重把最低 MEL->NV 进一步降到 0.11224，却损伤整体指标。这支持完成
冻结 2×2 因素组合：Run13 保持 Run12 注意力不变，只加入 Run11 的 MEL 权重，检验
两因素是否互补。

## 4. 决策

不继续 Run12。它已经正常早停且完整结束；任何继续写入都会破坏已完成实验和单因素
解释。Run12 作为注意力主效应实验原样保留，下一正式实验编号为 Run13。正式模型锁
仍未建立，禁止运行 test 或外部候选评价。
