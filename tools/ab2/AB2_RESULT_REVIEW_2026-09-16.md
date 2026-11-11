# AB2 训练结果复核（seed43）

`AB2_RESULT_STATUS=COMPLETE_VERIFIED`
`TRAINING_PERFORMED=TRUE`
`TEST_OR_EXTERNAL_USED=FALSE`

AB2 于 2026-09-16 00:45:02 +0800 正常结束，共 79 个可见 epoch。按预注册规则锁定 best-Joint / best-Macro-F1 的 visible epoch 49（bundle `epoch_0048`）。该 bundle fingerprint 为 `e50ab8dea5846fd01adfd621d1b4b025c7bbac278cec6d096e6e120d3e574738`，`trainer.pt` 与 `runtime_state.pt` SHA-256 分别为 `e643f5255f75a1524b81aac450b83b050d3100bb219a5a0e1d70bab9d3eea7af`、`364de092c597d2622d6f5aa293ad48f0273bf3a818ecce30fb82a0c93759885b`，复核一致。

锁定行：Binary IoU `0.76856`、Image Macro-F1 `0.51652`、Group Macro-F1 `0.59544`、Joint `0.67648`。相对同 seed Plain：IoU `+0.00479`，Image Macro-F1 `-0.03365`，Group Macro-F1 `-0.03267`，Joint `-0.01615`。相对完整 LGLA v1：IoU `-0.01007`、Group Macro-F1 `-0.05831`、Joint `-0.03698`。

固定 0.5/0.5 融合明显弱于完整 LGLA，说明样本级可学习融合门对分类和联合效用有实质贡献。AB2 的错误率/校准较好但核心分类效用较低，因此不能据此替代完整结构。
