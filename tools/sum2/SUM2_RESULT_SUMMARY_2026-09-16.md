# sum2：LGLA v1 扩展实验总复核

`SUM2_STATUS=COMPLETE`
`FINAL_ARCHITECTURE=LGLA_V1_GATE_BALANCE`
`MORE_LONG_TRAINING_REQUIRED=FALSE`
`THESIS_MATERIALS_READY=TRUE`

## Development 三 seed 主结果

| 指标 | Plain mean ± SD | LGLA v1 mean ± SD | 配对均值差 |
|---|---:|---:|---:|
| Binary IoU | 0.76364 ± 0.00037 | 0.77066 ± 0.00790 | +0.00702 |
| Image Macro-F1 | 0.54578 ± 0.00382 | 0.55945 ± 0.00229 | +0.01367 |
| Group Macro-F1 | 0.61492 ± 0.01158 | 0.63993 ± 0.01645 | +0.02501 |
| Group Recall | 0.61998 ± 0.02059 | 0.62103 ± 0.02098 | +0.00104 |
| Joint | 0.68524 ± 0.00647 | 0.70224 ± 0.01256 | +0.01700 |

Image Macro-F1、Group Macro-F1 和 Joint 均为 3/3 seed 提升；Binary IoU 为 2/3 seed 提升。参数量从 14,507,338 增至 14,540,748，增加 33,410（约 0.230%）。风险/校准指标整体变差：NLL、Brier、ECE 三 seed 均值分别增加约 0.15392、0.02369、0.02721；MEL→NV 与 malignant→benign 也未形成稳定改善。

## Seed43 结构对照

| 结构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint |
|---|---:|---:|---:|---:|
| Plain Run22 | 0.76377 | 0.55017 | 0.62811 | 0.69263 |
| AB1：未使用病灶引导的普通对照 run | 0.76719 | 0.54533 | 0.61436 | 0.68653 |
| AB2：固定 0.5/0.5 融合 | 0.76856 | 0.51652 | 0.59544 | 0.67648 |
| 完整 LGLA v1 | 0.77863 | 0.55681 | 0.65375 | 0.71346 |

完整结构的 Group Macro-F1 / Joint 分别比 AB1 高 0.03939 / 0.02693，比 AB2 高 0.05831 / 0.03698。对本科论文而言，这足以支持“病灶引导和样本级可学习融合均有贡献”的结构解释。

## 冻结测试结论

LGLA Test2 与 Plain Test1 的内部 634 张分割基本持平；受限 MEL/NV 子集 Accuracy 提高约 0.01190、Balanced Recall 提高约 0.00208，但 MEL Recall 低约 0.02083、NLL/Brier/ECE 更差。ISIC2018 增量 external 的 Binary IoU 低约 0.03898。

因此论文最稳妥的总述是：LGLA v1 在 development 三 seed 核心效用上稳定优于 Plain，并在冻结内部测试保持相当分割性能；但外部泛化优势没有得到补充测试支持，概率校准与 MEL 风险仍是局限。Test2 后不再继续训练或依据测试结果修改模型。
