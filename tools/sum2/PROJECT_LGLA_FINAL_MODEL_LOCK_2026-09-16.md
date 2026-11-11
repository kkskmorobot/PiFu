# LGLA v1 最终模型锁

`MODEL_LOCK_STATUS=LOCKED_BEFORE_TEST2`
`ARCHITECTURE=LGLA_V1_LESION_GUIDED_GLOBAL_LOCAL_LEARNED_GATE`
`SEEDS=42,43,44`
`MODEL_SELECTION_AFTER_TEST=FORBIDDEN`

## 锁定结论

Development 阶段按预注册规则锁定 LGLA v1 完整结构：病灶引导局部分支 + 全局分支 + 样本级可学习融合门。三个 seed 均使用各自训练期 best-Joint bundle，不用测试集选择模型。

| seed | run | visible epoch | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint |
|---:|---|---:|---:|---:|---:|---:|
| 42 | LG2 | 74 | 0.76284 | 0.56058 | 0.62173 | 0.68868 |
| 43 | LGX | 99 | 0.77863 | 0.55681 | 0.65375 | 0.71346 |
| 44 | LG3 | 97 | 0.77051 | 0.56096 | 0.64430 | 0.70459 |

与 Plain 的配对三 seed 均值相比，LGLA v1 的 Binary IoU、Image Macro-F1、Group Macro-F1 与 Joint 均提高；三 seed 的 Image Macro-F1、Group Macro-F1、Joint 均逐 seed 提高。风险/校准指标没有改善，因此论文结论限定为“核心分割与分类效用更好，但风险控制和概率校准仍是局限”。

## 冻结 checkpoint

- seed42：`epoch_0073`，fingerprint `16f9505ad452813aac9eb85e1b4cb176d077ff9d16a30ac20965ff3efbd87df4`
- seed43：`epoch_0098`，fingerprint `3a9aa4156828bd2e5af1b1f44a62929041cde5895d01ed995c8c5af823186ac0`
- seed44：`epoch_0096`，fingerprint `4a7c29f336b7ea28d8fbfe38e5e51cdff21f68ba418c78fddedbceb6956642aa`

Test2 只允许描述性冻结评估，不允许训练、再选择、阈值调整或根据结果返回 development 改模型。
