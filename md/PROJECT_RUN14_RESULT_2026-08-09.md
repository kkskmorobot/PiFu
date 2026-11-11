# PiFuV2 Run14 正式结果登记

日期：2026-08-09  
实验：`experiments/14_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVLogitMargin`

`RUN14_TRAINING_EXECUTION_GATE=PASS_NORMAL_EARLY_STOP`  
`RUN14_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN14_FEASIBILITY_GATE=FAIL_DEGENERATE_COLD_START_ONLY`  
`RUN14_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 训练和完整恢复

- 首次启动在 epoch 1 完成前意外断电；无完整 bundle，半成品已归档并 fresh start；
- 正式结果 `results.csv` 含连续可见 epoch 1~62 共 62 行，非有限值 0；
- 完成标记内部 `last_completed_epoch=61`；stopper `best_epoch=42`、patience=20、
  `possible_stop=true`，正常早停；
- 6 个 bundle `epoch_0000/0023/0047/0059/0060/0061` 全部通过大小、SHA-256、
  fingerprint 和完整状态验证；auto resume 指向 `epoch_0061`；
- 最新 runtime 含 live model、EMA、optimizer、scaler、scheduler、stopper、selection、
  Python/NumPy/Torch CPU/CUDA RNG 和 RandomSampler generator state；无中断请求；
- behavior fingerprint：
  `30e88fffb1eab5c59d7a3a1e0cbb458dcd2b020414c1011853941ce9fd456bc4`。

## 2. val-only 结果

- 最佳 binary IoU/Dice：可见 epoch 24，0.77703/0.87452；
- 最佳 image Macro-F1：可见 epoch 43，0.53792；
- 最佳 group Macro-F1/joint：可见 epoch 48，0.58802/0.67369；
- 最佳 joint 状态 MEL->NV=0.35714；最终状态 MEL->NV=0.36735；
- 唯一风险阈值内点是可见 epoch 1：MEL->NV=0.10204，但 binary IoU=0.53363、
  group Macro-F1=0.16923、joint=0.30051，明显是尚未学会任务的冷启动退化状态；
- 因此原 `best_feasible_joint=epoch_0000` alias 不能提升为正式模型锁。

## 3. 结论和选择协议修正

Run14 的 target-conditioned margin 位于八分类 softmax CE 内；当另外六类 logits 主导时，
MEL-vs-NV 梯度会被稀释。它没有保持 Run12 的整体指标，也没有在成熟模型阶段降低
MEL->NV。不继续已经正常早停的 Run14。

从 Run15 起，`feasible_joint` 除原 MEL support 和 MEL->NV 上限外，还必须满足
binary IoU>=0.70、group Macro-F1>=0.50。该非退化资格规则应用于模型锁和 alias，
但 early-stopping fitness 仍保持原 `joint - risk excess`，不改变训练轨迹。按此规则
复核 Run14，feasible 数量为 0。

Run14 所有结果、bundle 和断电归档原样保留；正式模型锁仍未建立，禁止 test/external。
