# Run31 结果复核与 Run32 决策（2026-09-08）

## Run31 完整性

- 中断恢复后完整训练至可见 epoch 100，CSV epoch 1–100 连续、无非有限值；
  `last_completed_epoch=99`，状态为正常训练完成。
- 最佳 Joint bundle 为零基 `epoch_0070`（可见 epoch 71）。
- 4 个保留 bundle（`0070/0097/0098/0099`）共 8 个权重/状态文件；实际大小与 SHA-256
  均和 manifest 一致，`_SUCCESS`、epoch 与 fingerprint 全部通过。
- fingerprint 为 `6200795f73c0adee421b1d0e0086fdfd432e314017e0c03c8465af47fc8ddc69`。
- 未使用 IMA++ test、ISIC2018 external 或 PAD-UFES-20。

## Run31 最佳 Joint 与同 seed43 Run22 配对

| 指标 | Run22 | Run31 | Run31 - Run22 |
|---|---:|---:|---:|
| Binary IoU | 0.76377 | 0.77605 | +0.01228 |
| Image Macro-F1 | 0.55017 | 0.50177 | -0.04840 |
| Group Macro-F1 | 0.62811 | 0.55803 | -0.07008 |
| Group Recall | 0.64033 | 0.54494 | -0.09539 |
| MEL→NV | 0.40816 | 0.36735 | -0.04081 |
| malignant→benign | 0.28502 | 0.26087 | -0.02415 |
| Joint | 0.69263 | 0.65807 | -0.03456 |

Run31 有 58 个 utility-eligible epoch，但正式 feasible checkpoint 为 0；唯一 risk-feasible 点属于
冷启动、utility 不合格。辅助头只带来小幅风险改善，却明显破坏 8 类与 group 效用，因此未通过
“风险实质改善且 Joint/Group Macro-F1 无明显损害”的预注册门。

## 决策

停止 Run31 MEL/NV 窄辅助头，不复现 seed42/44，也不依据当前 validation 调其权重或结构。
Run32 回到冻结的 Run22 plain seed43，只测试一个不同的层级因素：从同一 256 维隐藏描述符增加
恶性/良性二类训练期辅助头。辅助目标覆盖全部官方分类标签，benign=`[0,2,3,5,6]`、
malignant=`[1,4,7]`；因其监督密度远高于 Run31，权重预先固定为 `0.25`。主 8 类 logits、
validation 指标、选择规则和 inference argmax 不变。Run1–Run31 全部保留。
