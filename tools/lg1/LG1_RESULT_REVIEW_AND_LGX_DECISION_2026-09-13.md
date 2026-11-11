# lg1 结果复核与 lgx 决策（2026-09-13）

`LG1_RESULT_REVIEW_STATUS=COMPLETE`

`LG1_SCREENING_GATE=FAIL`

`NEXT_EXPERIMENT=LGX_SINGLE_GATE_BALANCE_REVISION`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 完整性复核

- 正式训练从 2026-09-12 21:28:28 +08:00 开始，于 2026-09-13 03:37:09 +08:00 正常完成；
- `results.csv` 共 100 行，可见 epoch 从 1 连续到 100，没有缺行；
- 完成标记的最后内部 epoch 为 99，对应可见 epoch 100，未提前停止；
- `best_joint_score` 与 `best_macro_f1` 都指向 `epoch_0083`，对应 CSV 可见 epoch 84；
- `best_binary_iou` 指向 `epoch_0027`，对应 CSV 可见 epoch 28；
- 最终 `latest` 指向 `epoch_0099`；所有保留 bundle 均有 `_SUCCESS`、manifest、trainer、runtime state；
- best-Joint bundle fingerprint 为 `ca47d032187061fe8107eb5152d4e3e28026426a6ea7e2df178a5fd146534732`；
- `epoch_0083/trainer.pt` SHA-256 为 `1c7652d079f1e1c148651e8a52e80bee0b433932225b43a5ee4857167efedd8f`；
- 训练和本次诊断均未使用 IMA++ test、ISIC2018 external 或 PAD-UFES-20。

## 权威 best-Joint 同行比较

| 指标 | Run22 Plain seed43 | lg1 LGLA v1 | lg1−Plain | 判定 |
|---|---:|---:|---:|---|
| Binary IoU | 0.76377 | 0.75967 | -0.00410 | 通过容差 |
| Image Macro-F1 | 0.55017 | 0.53761 | -0.01256 | 未通过 |
| Group Macro-F1 | 0.62811 | 0.61057 | -0.01754 | 未通过 |
| Group Recall | 0.64033 | 0.58858 | -0.05175 | 下降 |
| MEL→NV | 0.40816 | 0.33673 | -0.07143 | 风险方向改善 |
| malignant→benign | 0.28502 | 0.24638 | -0.03864 | 风险方向改善 |
| Joint | 0.69263 | 0.68105 | -0.01158 | 未通过 |

lg1 仅满足 Binary IoU 下降不超过 0.005；Group Macro-F1、Joint 与 Image Macro-F1 三个核心门均失败。
因此整体性能低于 Plain，不能进入 lg2/lg3 配对复现；它只能表述为 MEL→NV 与恶性转良性错误率的
定向改善，同时牺牲了总体分类和综合效用。

## 冻结 validation 门控诊断

使用 best-Joint bundle `epoch_0083` 进行仅 validation 的后验诊断：

- 局部门权：mean=`8.5977e-13`，population std=`3.4094e-12`，max=`6.0095e-11`；
- 全局门权：mean=`1.0`，std=`0.0`；
- P3/P4 注意力质量均值=`0.31914`，均有限；
- P3/P4 空注意力回退率均为 `0`；
- 全局/局部描述符余弦均值=`0.75039`，相对 L2 差异均值=`3.70027`；
- 门控末层 bias 已漂移到 `[-3.60156,+3.60156]`，进一步支持系统性门控偏置诊断。

注意力图本身没有空值或 NaN，局部描述符也不是全局描述符的恒等副本；失败机制是门控几乎完全关闭
局部分支，而不是分割路径退化。诊断文件位于正式结果目录的
`analysis/val_attention_diagnostics_v1/LG1_VAL_ATTENTION_DIAGNOSTICS.json`。

## 下一步决策

严格执行计划阶段 C2：只准备一次 `lgx`，不直接训练 lg2/lg3。lgx 的唯一改动是对有分类标签样本
加入系数 `0.01` 的批次均值门控-logit平衡正则；不加入 P5/GeM 或其他同步改动。完整预注册见
`tools/lgx/LGX_PROTOCOL_2026-09-13.md`。lgx 若仍不通过，停止 LGLA 继续训练路线。

