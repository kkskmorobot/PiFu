# Test1 一次性最终评估结果摘要

`TEST1_RESULT_STATUS=COMPLETE`
`MODEL_SELECTION_PERFORMED=FALSE`
`TRAINING_PERFORMED=FALSE`
`POST_TEST_DEVELOPMENT_CHANGES_PERMITTED=FALSE`

Test1 于 `2026-09-11 14:31:05 +0800` 完成。模型固定为 Plain YOLO26m P3/P4 partial-label 多任务架构，使用 Run18/22/24 三个预先锁定的 seed42/43/44 checkpoint；代表 seed 在测试前已固定为 seed43。推理规则为温度 `1.0` 的八类 argmax，分割阈值为 `0.5`，没有训练、调参、校准、阈值修改或事后 seed 选择。

## IMA++ 内部 test 分割（634 张）

| seed | Binary IoU | Binary Dice | Pixel Accuracy |
|---:|---:|---:|---:|
| 42 | 0.78844 | 0.88171 | 0.94706 |
| 43 | 0.78615 | 0.88027 | 0.94662 |
| 44 | 0.78182 | 0.87755 | 0.94507 |
| 三 seed 均值 ± 样本 SD | 0.78547 ± 0.00336 | 0.87984 ± 0.00211 | 0.94625 ± 0.00105 |

三 seed 的内部 test 分割结果较稳定，Binary IoU 的样本标准差约为 `0.00336`。

## IMA++ MEL/NV 标签子集分类（56 张）

该子集只有 MEL 16 张、NV 40 张。以下结果不是完整八类测试 Macro-F1，也不计算 Joint。

| seed | Accuracy | MEL Recall | NV Recall | Balanced Recall | MEL→NV | NV→MEL | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.78571 | 0.43750 | 0.92500 | 0.68125 | 0.56250 | 0.05000 | 0.63359 | 0.32420 | 0.12330 |
| 43 | 0.80357 | 0.43750 | 0.95000 | 0.69375 | 0.56250 | 0.05000 | 0.67214 | 0.31075 | 0.09659 |
| 44 | 0.75000 | 0.50000 | 0.85000 | 0.67500 | 0.50000 | 0.10000 | 0.85117 | 0.39374 | 0.19906 |
| 三 seed 均值 ± 样本 SD | 0.77976 ± 0.02728 | 0.45833 ± 0.03608 | 0.90833 ± 0.05204 | 0.68333 ± 0.00955 | 0.54167 ± 0.03608 | 0.06667 ± 0.02887 | 0.71897 ± 0.11610 | 0.34290 ± 0.04454 | 0.13965 ± 0.05316 |

最终测试揭示出清晰局限：NV recall 较高，但 MEL recall 只有约 `0.458`，MEL→NV 约为 `0.542`。这与 development 阶段未满足正式风险约束的结论一致，必须作为论文局限如实报告，不能在测试后重新优化。

预先固定的代表 seed43 的 95% 分层 bootstrap CI：Accuracy `[0.71429, 0.89286]`、MEL Recall `[0.18750, 0.68750]`、NV Recall `[0.87500, 1.00000]`、Balanced Recall `[0.56875, 0.81875]`、MEL→NV `[0.31250, 0.81250]`、NV→MEL `[0.00000, 0.12500]`。

## ISIC2018 增量 external 分割（89 张）

| seed | Binary IoU | Binary Dice | Pixel Accuracy |
|---:|---:|---:|---:|
| 42 | 0.65381 | 0.79067 | 0.93228 |
| 43 | 0.63596 | 0.77747 | 0.93027 |
| 44 | 0.61064 | 0.75825 | 0.92492 |
| 三 seed 均值 ± 样本 SD | 0.63347 ± 0.02169 | 0.77546 ± 0.01630 | 0.92916 ± 0.00380 |

External Binary IoU 均值比内部 test 低约 `0.15200`，且 seed 间波动更大，说明跨数据子域泛化仍有限。该集合仍属于 ISIC 相关域，只能称为增量 external segmentation，不能描述为独立医院临床验证。

## 审计与边界

- 正式结果以 `experiments/test1/results/result.json` 为唯一机器可读来源；SHA-256 为 `5825323ce93a21272b37ae4b0a84d819e48d9edcff6080ed580e942461dd5557`。
- 正式 fingerprint 为 `e6b1cfa834cc682922527cd1898ea6ea65a8f4999593bb70bba5e85c3fb20d42`。
- 每个 seed 的分类 CSV 均为 56 行；正式结果目录没有未归档 partial。
- 三个初始化/评估封装失败尝试已保留在 `experiments/test1/attempts`，均不属于正式结果。
- PAD-UFES-20 未使用；没有训练或模型选择。
- Test1 一次性授权已经消费，development 阶段永久关闭。
