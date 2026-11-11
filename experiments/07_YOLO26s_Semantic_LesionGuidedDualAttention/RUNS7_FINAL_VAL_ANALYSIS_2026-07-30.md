# Run7 正式训练与统一 val 最终分析（2026-07-30）

## 1. 训练完整性与早停

- `results.csv` 共 55 行，对应 epoch 1–55。
- 配置为 100 epochs、patience=20。
- 在线综合 fitness 最佳为 epoch 35；之后 20 轮未刷新，因此 epoch 55 正常早停，不是崩溃或中断。
- 图像 Macro-F1 最佳为 epoch 43；在线 foreground-mIoU 最佳为 epoch 41。
- `best.pt`、`best_macro_f1.pt`、周期权重、`last.pt` 和最终图表均完整。
- 55 轮中 foreground-mIoU 下限 0.518748 的通过次数为 0，所以没有 `best_feasible_macro_f1.pt`。
- 本次分析仅使用 val；未访问 test 指标。

## 2. 三个 Run7 Pareto 候选的统一 val

| checkpoint | 对应轮次 | foreground mIoU | Image Acc | Image Macro-F1 | Minority Recall | mel→nv | nv→mel | Lesion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best.pt` | 35 | 0.473481 | 0.793218 | 0.683146 | 0.607572 | 18 | 126 | 0.724177 |
| `best_macro_f1.pt` | 43 | 0.491179 | 0.815160 | 0.693490 | 0.519824 | 19 | 85 | 0.730219 |
| `epoch40.pt`（日志 epoch 41） | 41 | 0.502278 | 0.814495 | 0.683042 | 0.557548 | 19 | 96 | 0.734825 |

统一评估目录分别为：

- `val_unified_best_composite_epoch35`
- `val_unified_best_macro_epoch43`
- `val_unified_best_foreground_epoch41`

## 3. 与既有正式结果比较

| 模型 | foreground mIoU | Image Acc | Image Macro-F1 | Minority Recall | mel→nv | nv→mel | Lesion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run4 best | 0.538748 | 0.809176 | 0.704120 | 0.533498 | 18 | 84 | 0.724283 |
| Run6 best | 0.489603 | 0.805186 | 0.701557 | 0.629801 | 14 | 121 | 0.715276 |
| Run7 `best.pt` | 0.473481 | 0.793218 | 0.683146 | 0.607572 | 18 | 126 | 0.724177 |
| Run7 `best_macro_f1.pt` | 0.491179 | 0.815160 | 0.693490 | 0.519824 | 19 | 85 | 0.730219 |
| Run7 `epoch40.pt` | 0.502278 | 0.814495 | 0.683042 | 0.557548 | 19 | 96 | 0.734825 |

## 4. 结论

Run7 不能作为当前最终胜出模型：

- 三个候选都低于 foreground-mIoU 约束 0.518748；最接近的 epoch 41 仍低 0.016470。
- 主指标 Image Macro-F1 没有超过 Run4 或 Run6。
- Macro-F1 最佳点同时违反 minority Recall 与 mel→nv 约束。
- 综合分最佳点整体弱于 Run6，仅 lesion-level Macro-F1 有恢复。

但 Run7 不是毫无价值：

- epoch 41 的 lesion-level Macro-F1 为 0.734825，高于 Run4 的 0.724283 和 Run6 的 0.715276。
- epoch 43 的 Image Accuracy 为 0.815160，`nv→mel=85`，接近 Run4 的 84，明显优于 Run6 的 121。
- 这说明病灶引导聚合可能改善 lesion 级稳定性和 NV/MEL 平衡，但当前按增强图片训练的普通 CE 没有把收益转化为图像 Macro-F1 与分割提升。
- epoch 41 lesion Macro-F1 的 bootstrap 95% CI 为约 0.6722–0.7859，范围较宽，不能宣称相对 Run4 有可靠提升。

## 5. 后续决策

- 不从 `last.pt` 继续训练。epoch 35 到 55 期间 train image-cls loss 从 0.17154 降到 0.05116，而 val image-cls loss 从 0.64940 升到 0.99635，属于明确过拟合。
- Run7 应作为论文中的“提出结构、单 seed、未通过正式约束”消融结果保留。
- 下一实验固定 Run7 结构，优先只改变一个因素：lesion-balanced sampler。不要在同一次实验同时加入 sampler、balanced focal、风险代价和新增强。
- 若 sampler 单独有效，再单列下一实验验证 balanced softmax/focal；最终候选才进行多 seed 与 test。
