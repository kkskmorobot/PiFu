# Run19 结果复核与 Run20 决策

`RUN19_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN20_ADAPTIVE_RESIDUAL_ATTENTION`

Run19 于 2026-08-12 16:07 正常早停，完成可见 epoch 71；无异常，未使用 test 或外部数据，五个保留 bundle 的 manifest、大小、SHA-256 和 `_SUCCESS` 均通过。

## 最佳统一 val

| Run | best Binary IoU | best Image Macro-F1 | best Group Macro-F1 | best Joint | feasible |
|---|---:|---:|---:|---:|---:|
| Run18 plain | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |
| Run19 fixed BLG-MDA | 0.77358 | 0.54688 | 0.60160 | 0.68049 | 0 |

Run19 的最佳 joint 比 Run18 低 0.00197，image Macro-F1 低 0.01628，group Macro-F1 低 0.00863；Binary IoU 高 0.00044。注意力没有在当前形式下超过普通头。

## 同 epoch 诊断

- epoch 1–39：Run19 平均 joint +0.01051，group Macro-F1 +0.01638；
- epoch 40–71：Run19 平均 joint -0.00657，group Macro-F1 -0.01791；
- epoch 50–71：Run19 平均 joint -0.01003，group Macro-F1 -0.02307。

这表明模块并非完全无效，而是固定 0.5 病灶池化在后期形成过强、不可按尺度调整的偏置。因此 Run20 只把固定共享 mix 改为初始 0.05、上限 0.5 的 P3/P4/P5 可学习残差门，其他因素全部锁定。
