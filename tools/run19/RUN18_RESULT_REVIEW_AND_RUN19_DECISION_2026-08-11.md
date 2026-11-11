# Run18 结果复核与 Run19 决策

`RUN18_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN19_MATCHED_ATTENTION_ABLATION`

## Run18 完整性

- 完成标记：`TRAINING_COMPLETE_OR_NORMAL_EARLY_STOP`；
- 实际完成：100/100 epochs；
- test metrics：false；external metrics：false；
- 保留 bundles：epoch 49、94、97、98、99，manifest/大小/SHA-256/_SUCCESS 全部通过；
- aliases：best joint 与 best group 指向内部 epoch 49（可见 epoch 50），best binary 指向内部 epoch 94（可见 epoch 95）。

## 统一 val 对比

| Run | epochs | best binary IoU | best image Macro-F1 | best group Macro-F1 | best joint | feasible rows |
|---|---:|---:|---:|---:|---:|---:|
| Run16 BLG-MDA | 61 | 0.77389 | 0.54501 | 0.60160 | 0.68049 | 0 |
| Run17 plain / old early stop | 26 | 0.77029 | 0.49981 | 0.52180 | 0.62869 | 0 |
| Run18 plain / corrected lifecycle | 100 | 0.77314 | 0.56316 | 0.61023 | 0.68246 | 0 |

Run18 相对 Run17 的大幅改善证明旧早停确有问题。Run18 与 Run16 的 joint 只差 +0.00197，且训练时长不一致，不能得出注意力无效结论。

## 决策

不同时修改损失、采样、batch 或主干。Run19 只在 Run18 上恢复 Run16 的 BLG-MDA 注意力，并保留 Run18 的 100 epoch / 延迟效用门控早停，从而形成可写入论文的严格单因素消融对照。

本复核未运行 test 或外部评价。
