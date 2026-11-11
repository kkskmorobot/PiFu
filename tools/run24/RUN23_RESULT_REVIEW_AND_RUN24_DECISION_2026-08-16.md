# Run23 结果复核与 Run24 决策

`RUN23_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN24_PLAIN_SEED44_REPLICATION`

Run23 于 2026-08-16 18:16 正常早停。`results.csv` 覆盖 epoch 1~70；最佳 early-stop fitness 位于 CSV epoch 40，之后 30 个 epoch 未刷新，恰好在预注册的最早允许边界 epoch 70 停止。完成记录、别名和零基 latest bundle `epoch_0069` 相互一致。没有遗留 Python 训练进程，未运行 test 或外部候选评价。

五个保留 checkpoint bundle 均存在 `_SUCCESS`；`trainer.pt` 与 `runtime_state.pt` 的大小、SHA-256 和 fingerprint 全部通过。

## 两组同 seed 配对峰值

| seed | plain / ECA | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint | feasible |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Run18 plain | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |
| 42 | Run21 ECA | 0.77611 | 0.54432 | 0.60219 | 0.68285 | 0 |
| 43 | Run22 plain | 0.76999 | 0.55389 | 0.62811 | 0.69263 | 0 |
| 43 | Run23 ECA | 0.76674 | 0.55723 | 0.58762 | 0.66907 | 0 |

各列为该指标的独立 development-val 峰值，不保证来自同一 epoch。

配对差值：

- seed42（Run21−Run18）：Binary IoU `+0.00297`，Image Macro-F1 `−0.01884`，Group Macro-F1 `−0.00804`，Joint `+0.00039`；
- seed43（Run23−Run22）：Binary IoU `−0.00325`，Image Macro-F1 `+0.00334`，Group Macro-F1 `−0.04049`，Joint `−0.02356`。

两组的 Joint 与图像级分类方向不一致，但 Group Macro-F1 均下降。现有证据不支持宣称 ECA 稳定提升整体性能，并提示它可能伤害病灶组级分类；然而两组仍不足以稳定估计随机种子方差。

## Run24 决策

遵循 Run23 协议，在改变方法前补第三组配对。Run24 严格重复 plain 基线，只把 seed 从 43 改为 44；后续 Run25 才在相同 seed44 下加入与 Run21/Run23 完全相同的两个 P3/P4 ECA 门。

Run24 不修改数据、split、损失、采样、增广、batch、训练预算、延迟早停、风险阈值或模型选择规则。正式模型锁定前仍不得运行 test 或外部评价。
