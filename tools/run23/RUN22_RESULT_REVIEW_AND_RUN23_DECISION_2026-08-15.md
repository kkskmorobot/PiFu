# Run22 结果复核与 Run23 决策

`RUN22_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN23_P3P4_ECA_ONLY_SEED43_REPLICATION`

Run22 于 2026-08-15 16:12 正常早停。`results.csv` 覆盖 epoch 1~93，超过预注册的最早可见早停 epoch 70；完成记录的 `last_completed_epoch=92` 使用零基编号，与 CSV 的 93 个 epoch 一致。未发现遗留 Python 训练进程，没有运行 test 或外部候选评价。

五个保留 checkpoint bundle（最佳 Binary IoU、最佳 Joint/Macro-F1 和最后三个 epoch）均存在 `_SUCCESS`；`trainer.pt`、`runtime_state.pt` 的大小、SHA-256 与 manifest 全部一致，fingerprint 均为 `6a9cc96bd2d21f28b9c17ed745a50e76846f05d9c96153b2bad69f59c7257e82`。

## 统一 development val 峰值

| Run | seed | 结构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint | feasible |
|---|---:|---|---:|---:|---:|---:|---:|
| Run18 | 42 | P3/P4 plain GAP | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |
| Run21 | 42 | P3/P4 ECA + GAP | 0.77611 | 0.54432 | 0.60219 | 0.68285 | 0 |
| Run22 | 43 | P3/P4 plain GAP | 0.76999 | 0.55389 | 0.62811 | 0.69263 | 0 |

各列是对应指标独立峰值，不一定来自同一 epoch。Run22 最佳 Joint 位于 CSV epoch 60（bundle `epoch_0059`，bundle 使用零基 completed epoch），同一行 Binary IoU=0.76377、Image Macro-F1=0.55017、Group Macro-F1=0.62811、mel→nv=0.40816、Joint=0.69263。

## 判断

Run21 相对 Run18 的最佳 Joint 仅增加 0.00039；Run22 plain 仅改变 seed 后相对 Run18 已变化 +0.01017。随机种子波动明显大于单个 seed 下观察到的 ECA 表面增益，因此不能据 seed42 单次结果宣称 ECA 有效或无效。

Run18、Run21、Run22 均无风险合规 checkpoint。Run22 只有 1 个 epoch 满足 mel→nv 风险门，但没有与效用门同时形成 feasible checkpoint；不得为追求合规而临时改阈值、损失或选择规则。

## Run23 决策

按原预注册顺序训练 Run23：严格重复 Run21 的 P3/P4 ECA-only 结构，但使用与 Run22 相同的 seed=43。相对 Run22 唯一训练因素是加入两个 identity-initialized ECA 门；数据、模型权重、损失、采样、增广、优化器、训练预算、延迟早停、统一 val、风险阈值和 checkpoint 合同保持不变。

Run23 完成后，先比较两组同 seed 配对差值（Run21-Run18、Run23-Run22），再决定是否需要第三个 seed。正式模型锁定前仍不得运行 test 或外部评价。
