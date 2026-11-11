# Run33 结果复核与 Run34 决策（2026-09-10）

`RUN33_REVIEW_STATUS=PASS`

## Run33 完整性

- `results.csv` 共 100 行，可见 epoch 1～100 连续，无 NaN/Inf。
- 最佳 Joint alias 为零基 `epoch_0074`，即可见 epoch 75。
- 6 个完整 checkpoint bundle 的大小、SHA-256、`_SUCCESS` 和 fingerprint 全部匹配。
- 正常完成 100 epoch；中途恢复来自完整 bundle，不影响完整性。
- `test_metrics_used=false`，`external_metrics_used=false`。

## 与 Run22 seed43 Plain 配对比较

| 指标 | Run22 | Run33 | Run33 - Run22 |
|---|---:|---:|---:|
| Binary IoU | 0.76377 | 0.76789 | +0.00412 |
| Image Macro-F1 | 0.55017 | 0.54022 | -0.00995 |
| Group Macro-F1 | 0.62811 | 0.61167 | -0.01644 |
| Group Recall | 0.64033 | 0.60453 | -0.03580 |
| MEL→NV | 0.40816 | 0.38776 | -0.02040 |
| malignant→benign | 0.28502 | 0.28502 | 0.00000 |
| Joint | 0.69263 | 0.68534 | -0.00729 |

Run33 同时跌破 Group Macro-F1 与 Joint 守门线，且没有任一材料性收益达到预注册阈值，因此 CBAM 复现门判定为失败。不进行 seed42/44 CBAM 复现，不修改 CBAM 结构后继续追试。

## 模型锁定决策

三 seed 的最佳 Joint row 汇总表明，Plain 相对 ECA 和 ECA+spatial7 具有更高的平均 Image Macro-F1、Group Macro-F1、Group Recall 和 Joint，并且离散程度更小。ECA+spatial7 只在 MEL→NV 平均值上略有改善，但 seed42 明显失败，不足以作为稳定主线。

因此锁定：

- 最终训练架构：Plain YOLO26m P3/P4 partial-label 多任务基线。
- 三 seed 开发证据：Run18/Run22/Run24。
- checkpoint 规则：每个 seed 使用其已锁定的 best-Joint bundle。
- 停止新的注意力组合和训练期风险损失实验。

## 下一步

Run34 为 development-only 的三 seed 固定代价决策研究，仅复用 Run18/22/24 的冻结 checkpoint，不训练、不调参、不访问 test/external/PAD-UFES-20。
