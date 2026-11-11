# Run21 结果复核与 Run22 决策

`RUN21_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN22_PLAIN_SEED43_REPLICATION`

Run21 于 2026-08-14 15:08 正常完成全部 100 epochs；无异常、中断、test 或外部评价。5 个保留 bundle 的 manifest、大小、SHA-256 和 `_SUCCESS` 全部通过。

## 统一 val 峰值

| Run | seed | 结构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint | feasible |
|---|---:|---|---:|---:|---:|---:|---:|
| Run18 | 42 | P3/P4 plain GAP | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |
| Run21 | 42 | P3/P4 ECA + GAP | 0.77611 | 0.54432 | 0.60219 | 0.68285 | 0 |

Run21 的最佳 Joint 仅高 0.00039，Binary IoU 高 0.00297，但 Image Macro-F1 低 0.01884、Group Macro-F1 低 0.00804。该差值不足以排除单一随机种子波动，不能据此宣称 ECA 稳定提升整体性能。

## 配对重复决策

- Run22 严格重复 Run18 plain，仅把 seed 从 42 改为 43；
- 后续 Run23 严格重复 Run21 ECA-only，同样使用 seed 43；
- Run22/Run23 在 seed 43 下仍只差两个 identity-initialized P3/P4 ECA 模块。

此顺序先建立第二个无模块基线，再训练对应加模块版本，形成论文需要的配对消融证据。Run22 不新增注意力、不改损失、不调阈值，也不读取 test。
