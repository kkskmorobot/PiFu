# AB1 训练结果复核（seed43）

`AB1_RESULT_STATUS=COMPLETE_VERIFIED`
`TRAINING_PERFORMED=TRUE`
`TEST_OR_EXTERNAL_USED=FALSE`

AB1 于 2026-09-15 18:47:55 +0800 正常结束，共 98 个可见 epoch。按预注册规则锁定 best-Joint / best-Macro-F1 的 visible epoch 68（bundle `epoch_0067`）。该 bundle 的 fingerprint 为 `5bc2f059c8d85d45aa6addd4496ad706ca7de03144cf9fe28d9aa0ad956ce122`，`trainer.pt` 与 `runtime_state.pt` 的 SHA-256 分别为 `af054576f32f8dff8b72d07832ec1585a95b6370a3f57b396903e4257824fd32`、`72b07f03bda008bfea1c8dec7b474db919dd6690da6d1be4f0a3fd651e12663c`，复核一致。

锁定行：Binary IoU `0.76719`、Image Macro-F1 `0.54533`、Group Macro-F1 `0.61436`、Joint `0.68653`。相对同 seed Plain：IoU `+0.00342`，Image Macro-F1 `-0.00484`，Group Macro-F1 `-0.01375`，Joint `-0.00610`。相对完整 LGLA v1：IoU `-0.01144`、Group Macro-F1 `-0.03939`、Joint `-0.02693`。

论文中可把 AB1 写成“未使用病灶引导的普通对照 run”。结果表明仅保留全局/局部结构而移除病灶引导，无法复制完整 LGLA 的分类和联合效用提升，支持病灶引导分支的有效性。
