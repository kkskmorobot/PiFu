# PiFuV2 Run15 正式结果登记

日期：2026-08-09  
实验：`experiments/15_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVPairwiseBoundary`

`RUN15_TRAINING_EXECUTION_GATE=PASS_NORMAL_EARLY_STOP`  
`RUN15_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN15_FEASIBILITY_GATE=FAIL_NO_NON_DEGENERATE_FEASIBLE_CHECKPOINT`  
`RUN15_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 训练与完整恢复

- `results.csv` 含连续可见 epoch 1~58 共 58 行，非有限值 0；
- 完成标记内部 `last_completed_epoch=57`，对应可见 epoch 58；
- stopper `best_epoch=38`、`best_fitness=0.51397`、patience=20、
  `possible_stop=true`，正常早停；
- 5 个完整 bundle `epoch_0020/0031/0055/0056/0057` 均通过文件大小、SHA-256、
  fingerprint 和完整状态验证；auto resume 指向 `epoch_0057`；
- 最新 runtime 包含 live model、EMA、optimizer、scaler、scheduler、stopper、selection、
  Python/NumPy/Torch CPU/CUDA RNG 与 RandomSampler generator state；无中断请求；
- behavior fingerprint：
  `f2e38bd9fd0422280de8c1e8ac3520d5bf6f0843efaef343d7e0601f3a43b978`。

## 2. val-only 统一结果

- 最佳 binary IoU/Dice：可见 epoch 21，0.77587/0.87379；
- 最佳 image Macro-F1：可见 epoch 49，0.54019；
- 最佳 group Macro-F1/joint：可见 epoch 32，0.57479/0.66724；
- 最佳 joint 状态 MEL->NV=0.29592、malignant->benign=0.21256；
- 最低 MEL->NV 出现在可见 epoch 1，为 0.01020，但 binary IoU=0.63758、
  group Macro-F1=0.11139、utility eligible=0，属于冷启动退化点；
- 最终可见 epoch 58：group Macro-F1=0.55691、joint=0.65425、MEL->NV=0.28571；
- 非退化选择协议下 feasible 数量为 0，`best_feasible_joint` alias 不存在。

## 3. 结论

Run15 的 direct pairwise boundary loss 没有超过 Run12：Run12 最佳 group Macro-F1/joint
为 0.59969/0.68258，Run15 降至 0.57479/0.66724；成熟阶段 MEL->NV 也没有形成
可行点。Run15 已正常早停，不继续训练，也不把 pairwise loss 带入下一实验。

下一步回到 Run12 的 BLG-MDA + unweighted CE，只把官方预训练模型规模从
YOLO26s-sem 提升到 YOLO26m-sem，作为 Run16 唯一训练因素。Run15 所有结果和 bundle
原样保留；正式模型锁仍未建立，禁止 test/external 评价。
