# Run20 结果复核与 Run21 决策

`RUN20_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN21_P3P4_ECA_ONLY`

Run20 于 2026-08-13 13:44 正常完成全部 100 epochs；无异常、中断、test 或外部评价。5 个保留 bundle 的 manifest、大小、SHA-256 和 `_SUCCESS` 全部通过。

## 统一 val 峰值

| Run | 结构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Joint | feasible |
|---|---|---:|---:|---:|---:|---:|
| Run18 | P3/P4 plain GAP | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |
| Run19 | P3/P4/P5 ECA + fixed BLG | 0.77358 | 0.54688 | 0.60160 | 0.68049 | 0 |
| Run20 | P3/P4/P5 ECA + adaptive residual BLG | 0.77058 | 0.54239 | 0.57874 | 0.66463 | 0 |

Run20 相对 Run18 的最佳 joint 低 0.01783，group Macro-F1 低 0.03149，image Macro-F1 低 0.02077，Binary IoU 低 0.00256。

## 注意力门诊断

Run20 三个病灶引导 mix 从 `[0.05, 0.05, 0.05]` 学到：

- 最佳 Binary IoU bundle epoch 17：约 `[0.080, 0.092, 0.089]`；
- 最佳 joint bundle epoch 52：约 `[0.097, 0.119, 0.123]`；
- 最终 bundle epoch 99：约 `[0.106, 0.133, 0.138]`。

门参数确实获得梯度并持续增强病灶聚焦，所以 Run20 失败不是“门没有学动”。现有结果无法区分 ECA 通道注意力的净效应与空间病灶池化/P5 描述符的负效应。

## Run21 决策

以当前最强 Run18 为基线，只在其原 P3/P4 特征上加入 identity-initialized ECA，再执行同样的 global-average pooling：不使用病灶 mask、不使用 P5、不改变语义分割分支。这样 Run21 只检验通道注意力本身，形成论文所需的清晰模块消融。
