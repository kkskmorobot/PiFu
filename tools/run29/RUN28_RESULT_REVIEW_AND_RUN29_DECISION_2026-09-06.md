# Run28 结果复核与 Run29 决策

## Run28 完整性与最佳结果

- CSV 可见 epoch 1–100 连续，非有限值为 0；79 个 epoch 通过 utility floor；正式 feasible 为 0。
- 最佳 Joint bundle：零基 `epoch_0073`（可见 epoch 74）。
- Binary IoU `0.75894`，Image Macro-F1 `0.55139`，Group Macro-F1 `0.62621`，
  Group Recall `0.59894`，MEL→NV `0.34694`，malignant→benign `0.28502`，Joint `0.68939`。
- 五套完整 bundle（`0031/0073/0097/0098/0099`）共 10 个 checkpoint 文件；大小、
  SHA-256、`_SUCCESS` 与 fingerprint 全部复核通过。
- 未使用 test 或 external。

## 同 seed43 配对

Run28 − Run23：IoU `-0.00287`，Image Macro-F1 `+0.00094`，Group Macro-F1
`+0.03859`，Group Recall `+0.03550`，MEL→NV `-0.06122`，Joint `+0.02032`。

## 三 seed 空间门结论

空间门相对同 seed ECA-only 的三 seed 平均改变量：IoU `-0.00193`，Image Macro-F1
`+0.00529`，Group Macro-F1 `+0.00226`，Group Recall `+0.01356`，MEL→NV
`-0.06122`，Joint `+0.00015`。seed42 的 Joint 和 Group Macro-F1 明显反向，说明风险方向
有一致改善，但整体效用增益不稳定且接近零。

结论：不锁定 ECA+spatial7，不继续注意力主线。Run29 回到 plain seed43 基线，只测试一个
不增加模型参数的恶性组层级辅助 NLL。Run1–Run28 全部保留。

