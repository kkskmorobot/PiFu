# Val1 结果复核与最终决策（2026-09-11）

`VAL1_STATUS=COMPLETE`
`VAL1_ACCEPTANCE_GATE=FAIL`
`FINAL_DECISION=RETAIN_ARGMAX_AND_STOP_DECISION_RULE_TUNING`
`TRAINING_PERFORMED=FALSE`
`TEST_METRICS_USED=FALSE`
`EXTERNAL_METRICS_USED=FALSE`
`PAD_UFES_20_USED=FALSE`

## 命名与 provenance

本任务是冻结模型的 development-only validation，不产生新模型权重，因此规范名称定为 **Val1**，不占用新的训练 Run 编号。

它在预注册和执行时使用了历史名称 Run34。完成后，用户将 `tools\run34` 的全部内容移动到 `tools\val1` 并删除原目录。为保持预注册配置 SHA-256、fingerprint、完整结果和两次失败 partial 的审计链不被静默改写，冻结文件继续保留原来的 Run34 文件名和内部历史路径：

- 当前工具目录为 `tools\val1`，原始工具目录为 `tools\run34`；
- `experiments\34_YOLO26m_Plain_FixedCostDecision_ThreeSeed_DevelopmentOnly`；
- `RUN34_RESULT.json` 及两个 `*.partial-*` 失败目录。

从本文件开始，论文、路线图和后续汇总统一将该任务称为 Val1；引用旧文件名或 `experiments\34_...` 路径时注明“历史预注册名 Run34”。这是一项完成后的透明归类，不改变已经发生的预注册与执行事实。旧入口因目录搬迁后不再作为可重复执行入口使用；正式结果由结果哈希和本 provenance 文件追溯。

## 完整性复核

- 正式结果状态为 `COMPLETE`，完成时间为 `2026-09-11 12:43:37 +0800`。
- 预注册配置 SHA-256：`2952bb706ee8eacd24d75348be2d0dc1a8e8df6d7204a64fd0cc932ee15dbf0d`。
- 评估 fingerprint：`04a47ea9b7c8d35eb8f1cd047d0bb53eb037b2953c7216002a9f34b82e5d9ac2`。
- `RUN34_RESULT.json` SHA-256：`004639e58121162cd84156388494823109261102a65767e0924f01811f737a04`。
- 三个 seed 均使用对应 bundle 中 `runtime_state.pt::training_state.ema` 的无损 fp32 EMA，采用训练期未融合、CUDA autocast 验证路径。
- 每个 seed 均输出 1,488 行 image prediction 与 587 行 group prediction；六个 CSV 均可完整解析。
- 三个 seed 的 argmax 复现误差均低于预注册容差 `5e-4`。
- 未创建 optimizer、未 backward、未修改模型权重，未读取 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
- 两次失败 partial 原样保留，未伪装为完成结果。

## 预注册比较结果

以下差值均为“固定代价规则减原始 argmax”：

| seed / 来源 | Group Macro-F1 | Group Recall | MEL→NV | malignant→benign | Joint |
|---|---:|---:|---:|---:|---:|
| seed42 / Run18 | -0.06699 | -0.06353 | -0.24490 | -0.16908 | -0.03855 |
| seed43 / Run22 | -0.07780 | -0.09009 | -0.23469 | -0.14976 | -0.04431 |
| seed44 / Run24 | -0.00194 | -0.00943 | -0.14286 | -0.12077 | -0.00109 |
| 三 seed 平均 | -0.04891 | -0.05435 | -0.20748 | -0.14654 | -0.02798 |

风险方向的四项条件通过：每个 seed 的两项风险均未增加，且三 seed 平均 MEL→NV 与 malignant→benign 的降幅超过预注册门槛。

总体效用保护门未通过：平均 Group Macro-F1 下降超过 `0.03`，平均 Joint 下降超过 `0.02`，并且 seed42、seed43 的 Group Macro-F1 下降超过任一 seed 可接受的 `0.06`。

## 决策

固定代价规则虽明显减少恶性误判，但其收益由不可接受且不稳定的总体分类效用损失换取，因此予以拒绝。最终推理规则保持原始八类 argmax，温度为 `1.0`；不得在同一 development validation 上继续修改代价、阈值或规则。

当前仍无满足 `MEL→NV <= 0.1077844311` 的正式 feasible checkpoint。该约束未满足必须作为论文局限报告，不能把 best-Joint checkpoint 表述为临床风险合格模型。

按根目录路线图，训练期与固定代价决策分支均已关闭，不准备 Run35。下一阶段仅进行最终研究候选锁定、development 结果汇总和论文图表整理；任何一次性 test/external 评估都必须另行预注册、通过发布门并取得用户新的明确确认。
