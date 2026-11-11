# Run26 结果复核与 Run27 决策

`RUN26_RESULT_REVIEW=PASS`

`NEXT_EXPERIMENT_DECISION=RUN27_P3P4_ECA_IDENTITY_SPATIAL_ATTENTION_SEED42`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## Run26 完成状态

- Run26 于 2026-09-04 正常早停完成；`results.csv` 的可见 epoch 1～71 连续，非有限值为 0。
- 最佳联合 bundle 为零基 `epoch_0040`，对应可见 CSV epoch 41。
- 同一 checkpoint 的 Binary IoU=0.76971、Image Macro-F1=0.55333、Group Macro-F1=0.63762、Group Macro-Recall=0.65132、MEL→NV=0.33673、Joint=0.70056。
- 5 个保留 bundle 的 `_SUCCESS`、字节数、SHA-256 和 fingerprint 均复核通过。
- 风险可行 epoch 为 0，因此该 bundle 仅是 development-val 诊断点，不能锁为正式模型。

## 受控结果与证据缺口

Run26 相对同 seed44 的 Run25 ECA-only，在各自最佳 joint 同一行上：Binary IoU +0.00530、Image Macro-F1 +0.02572、Group Macro-F1 +0.01391、Group Macro-Recall +0.06874、MEL→NV -0.11225、Joint +0.01007。该结果支持复现空间门，但只有一个 seed，不能证明稳定收益。

seed42 的现成直接基线是 Run21 ECA-only。Run21 最佳 joint bundle 为零基 `epoch_0040`（可见 epoch 41），同一行 Binary IoU=0.77431、Image Macro-F1=0.53745、Group Macro-F1=0.60219、Group Macro-Recall=0.60337、MEL→NV=0.40816、Joint=0.68285。其正式训练 fingerprint 为 `b7b853c4284037a6d51d0e66e10cd0302d167bf4ab30c7b32f512f6f137fb1a9`。

## Run27 决策

Run27 以 Run21 为直接基线，seed 固定为 42。唯一新增因素是在 P3、P4 的 identity-initialized ECA 输出后各增加一个无 bias 的 7×7 空间卷积门，使用通道 mean/max 描述与 `2*sigmoid` 缩放；空间卷积全零初始化，使初始门严格为 1。

Run27 不使用预测掩码、不增加 P5、不修改语义分支，也不改变数据、loss、优化器、batch、训练预算、增强、早停、风险阈值、checkpoint 合同或模型选择规则。正式结果写入新的 experiments 27 目录，不回写 Run21 或 Run26。

Run27 完成后只做 development-val 配对比较；在 Run28 seed43 完成前不锁定注意力主线。正式模型锁定及用户重新授权前，禁止运行 IMA++ test、ISIC 2018 external 或 PAD-UFES-20。
