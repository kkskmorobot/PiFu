# Run27 结果复核与 Run28 决策

`RUN27_RESULT_REVIEW=PASS_NORMAL_EARLY_STOP`

`RUN27_CHECKPOINT_INTEGRITY=PASS`

`NEXT_EXPERIMENT_DECISION=RUN28_SAME_SPATIAL7_SEED43_REPLICATION`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## Run27 完成与完整性

- Run27 于 2026-09-05 17:49:15 +0800 正常早停完成。
- `results.csv` 有连续可见 epoch 1～91 共 91 行，非有限值为 0。
- 风险惩罚后的早停适应度在可见 epoch 61 最佳，之后连续 30 个 epoch 未改善，于可见 epoch 91 正常停止；不是崩溃。
- 保留的 5 个 bundle 为 `epoch_0070`、`epoch_0074`、`epoch_0088`、`epoch_0089`、`epoch_0090`。
- 5 个 bundle 的 `_SUCCESS`、文件字节数、SHA-256 和 fingerprint `85880df0bfb882373d0ffe89fcd2f65b5c3c9bf50ecd9b2f5bf1a4c0a57a10fa` 全部复核通过。
- 未运行 test 或 external。

## Run27 指标

最佳 joint bundle 为零基 `epoch_0074`，对应可见 CSV epoch 75；同一行指标：

| 指标 | Run27 |
|---|---:|
| Binary IoU | 0.76610 |
| Binary Dice | 0.86756 |
| Image Macro-F1 | 0.52667 |
| Group Macro-F1 | 0.55647 |
| Group Macro-Recall | 0.53981 |
| MEL→NV | 0.39796 |
| Malignant→Benign | 0.29469 |
| NLL | 0.87122 |
| Brier | 0.39562 |
| ECE | 0.10159 |
| Joint | 0.65292 |

共有 63 个 utility-eligible epoch，但 risk-feasible 和完整 feasible epoch 均为 0。成熟阶段最低 MEL→NV=0.36735，仍远高于 0.1077844311，因此不能锁为正式模型。

## Run27−Run21 严格配对差值

以下均取各自 `best_joint_score` 的同一 CSV 行：

| 指标 | 差值 |
|---|---:|
| Binary IoU | -0.00821 |
| Image Macro-F1 | -0.01078 |
| Group Macro-F1 | -0.04572 |
| Group Macro-Recall | -0.06356 |
| MEL→NV | -0.01020（略好） |
| Joint | -0.02993 |

Run27 与 seed44 Run26 的正向结果方向冲突，证明空间门尚无稳定证据。

## 决策

不根据 Run27 的负向结果临时修改架构或 loss。这样做会成为看过 seed42 development val 后的事后调参，并破坏原定的三 seed 因果比较。

Run28 按原计划使用 seed43，以 Run23 ECA-only 为直接基线，只增加与 Run26/Run27 完全相同的 P3/P4 identity-initialized spatial7 门。Run23 正式 fingerprint 为 `8837f6958bf63820732ad852367dd2d638927fcdd9de0d0a96c039c2cdf55c13`；其最佳 joint 同一行 Binary IoU=0.76181、Image Macro-F1=0.55045、Group Macro-F1=0.58762、Group Macro-Recall=0.56344、MEL→NV=0.40816、Joint=0.66907。

Run28 完成后立即汇总三 seed；若预注册稳定收益判据不满足，停止该注意力架构方向，再以新的单因素 run 单独研究风险 loss。正式模型锁定前继续禁止 IMA++ test、ISIC 2018 external 和 PAD-UFES-20。
