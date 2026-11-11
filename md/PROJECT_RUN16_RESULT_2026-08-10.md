# Run16 正式训练结果与后续决策

`PROJECT_RUN16_RESULT_STATUS=COMPLETE_NORMAL_EARLY_STOP`

## 完成性审计

- 正式结果目录：`experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention/results/run16_yolo26m_binary8cls_binary_lesion_guided_attention_seed42`；
- `RUN16_TRAINING_COMPLETE.json` 记录正常完成或正常早停；内部 `last_completed_epoch=60`，对应 CSV 可见 epoch 61；
- `results.csv` 共 61 行，epoch 连续且无非有限值；
- 训练曾从完整 `epoch_0041` bundle 自动恢复，随后正常早停；
- 完整且通过大小、SHA-256、指纹与 `_SUCCESS` 校验的 bundle 共 5 个：epoch 35、54、58、59、60；
- 最终自动恢复点是 `epoch_0060`；行为指纹为 `2433474d4f826d8fcfed94e946eb9facf54d4ac13c20fdc5e2720855ddf18e27`；
- bundle 包含 live model、EMA、optimizer、scaler、scheduler、stopper、selection、Python/NumPy/Torch CPU/CUDA RNG 与 RandomSampler generator；
- 未运行 test 或外部候选评价。

## 统一 val 结果

- 最佳 binary IoU / Dice：0.77389 / 0.87254（可见 epoch 55）；
- 最佳 image Macro-F1：0.54501（epoch 37）；
- 最佳 group Macro-F1 / joint：0.60160 / 0.68049（epoch 36）；
- 最佳 joint checkpoint 的 MEL→NV：0.37755；
- 满足 binary IoU≥0.70 且 group Macro-F1≥0.50 后的最低 MEL→NV：0.34694（epoch 41，group Macro-F1 0.59534，joint 0.67794）；
- 最终 epoch 61：group Macro-F1 0.55262，joint 0.65374，MEL→NV 0.36735；
- 没有 checkpoint 满足预注册的 MEL→NV 上限 0.1077844311，`best_feasible_joint` 不存在。

## 结论

Run16 是完整、可复现且具有有效效用的正式训练，不废弃。其 group Macro-F1 0.60160 是 Run10–16 中最高，足以作为毕业论文的 YOLO26m + BLG-MDA 注意力版本；但风险约束未通过，且在没有同一 YOLO26m 无注意力基线前，不能把结果归因于注意力模块。

下一步正式 Run17 使用相同 YOLO26m、数据、训练和 val 协议，只移除注意力模块，以补齐严格消融对照。若 Run17 优于 Run16，Run16 仍作为负向消融结果保留并如实报告，不做事后删除。

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`
