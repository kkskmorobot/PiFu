# Run30 结果复核与 Run31 决策（2026-09-07）

## Run30 完整性

- 正常早停完成，CSV 可见 epoch 1–93 连续、无非有限值；`last_completed_epoch=92`。
- 最佳 Joint bundle 为零基 `epoch_0062`（可见 epoch 63）。
- 4 个完整 checkpoint bundle（`0062/0090/0091/0092`）共 8 个权重文件；大小与 SHA-256
  均和 manifest 一致，`_SUCCESS` 与 fingerprint 均通过。
- 未使用 IMA++ test、ISIC2018 external 或 PAD-UFES-20。

## Run30 最佳 Joint 与同 seed43 Run22 配对

| 指标 | Run22 | Run30 | Run30 - Run22 |
|---|---:|---:|---:|
| Binary IoU | 0.76377 | 0.77119 | +0.00742 |
| Image Macro-F1 | 0.55017 | 0.53682 | -0.01335 |
| Group Macro-F1 | 0.62811 | 0.59698 | -0.03113 |
| Group Recall | 0.64033 | 0.60039 | -0.03994 |
| MEL→NV | 0.40816 | 0.30612 | -0.10204 |
| malignant→benign | 0.28502 | 0.21739 | -0.06763 |
| Joint | 0.69263 | 0.67851 | -0.01412 |

Run30 的风险方向改善明显，但 Joint 与 Group Macro-F1 同时受到明显损害，且正式 feasible
checkpoint 仍为 0。唯一 `risk_feasible` 点属于冷启动、utility 不合格，不能视为可用模型。

## 决策

Run30 未通过其预注册推进门。停止该 focal risk loss，不复现 seed42/44，也不根据当前 validation
结果调 gamma/weight。Run31 转向预先列入路线图的层级辅助任务：以 Run22 为基线，只新增训练期
MEL/NV 二类辅助头；主 8 类预测和推理保持不变。Run1–Run30 全部保留。
