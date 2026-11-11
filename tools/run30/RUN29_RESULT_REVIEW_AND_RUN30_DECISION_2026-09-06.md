# Run29 结果复核与 Run30 决策

## 完整性

- 正常完成 100 epochs；CSV 可见 epoch 1–100 连续，非有限值 0。
- 70 个 utility-eligible epoch；risk-feasible 与完整 feasible 均为 0。
- 6 套完整 bundle、12 个 checkpoint 文件的大小、SHA-256、`_SUCCESS` 与 fingerprint 全部通过。
- 未使用 test 或 external。

## 最佳 Joint 与 Run22 同 seed43 配对

Run29 最佳 Joint bundle 为零基 `epoch_0035`（可见 epoch 36）：Binary IoU `0.77342`、
Image Macro-F1 `0.50836`、Group Macro-F1 `0.57909`、Group Recall `0.55161`、
MEL→NV `0.39796`、malignant→benign `0.28019`、Joint `0.66924`。

Run29 − Run22：IoU `+0.00965`，Image Macro-F1 `-0.04181`，Group Macro-F1
`-0.04902`，Group Recall `-0.08872`，MEL→NV `-0.01020`，malignant→benign
`-0.00483`，Joint `-0.02339`。成熟 utility-eligible 阶段最低 MEL→NV 仅到 `0.37755`。

## 判定

Run29 未满足预注册的“不损害 Group Macro-F1 且 Joint 不低于 Run22”条件。对全部标签对称施加
恶性/良性组 NLL 的效用代价显著，而风险收益很小，因此停止该机制，不做 seed42/44 复现。

Run30 仍使用 Run22 plain seed43，只测试一个更窄的 true-MEL→NV posterior focal risk loss；
不修改结构、数据或推理。

