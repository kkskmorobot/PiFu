# PiFuV2 Run10 正式结果登记

日期：2026-08-05  
实验：`experiments/10_YOLO26s_Binary8Class_PartialLabel_Baseline`

`RUN10_TRAINING_EXECUTION_GATE=PASS`  
`RUN10_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN10_FEASIBILITY_GATE=FAIL_NO_FEASIBLE_CHECKPOINT`  
`RUN10_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 已核实事实

- 2026-08-05 12:48:50 +0800 启动，16:41:16 +0800 正常完成；
- `results.csv` 含连续 epoch 1~100 共 100 行，非有限值为 0；
- 完成标记的 `last_completed_epoch=99`，即零基索引的第 100 个完成 epoch；
- epoch 43、44、97、98、99 的 5 个保留 bundle 均通过项目验证器的文件大小、
  SHA-256、fingerprint 和必需状态字段检查；auto resume 指向 epoch 99；
- Run10 behavior fingerprint：
  `a09e0cfb4669e06ff8ff67036699253282bbaa619c9bba28eb3e79bfb7445d9a`；
- 没有运行 test 或 external evaluator。

## val-only 结果

- 最佳 binary IoU：日志 epoch 44，0.78131；Dice 0.87723；
- 最佳 group Macro-F1 / joint：日志 epoch 45，0.59426 / 0.68132；
- 最佳 image Macro-F1：日志 epoch 56，0.54293；
- 100 个 epoch 中 feasible 数量为 0；
- 全程最低 mel->nv rate 为 0.32653；最佳 joint 状态为 0.43878；
- 冻结上限为 0.1077844311，因此 `best_feasible_joint` alias 不存在。

## 结论

Run10 是可复现、可恢复且训练完整的新任务基线，但按预注册协议属于约束失败。
`best_joint_score` 可作为诊断引用，不能被提升为正式锁定模型；不得利用 test 或
PAD/ISIC2018 external 结果改变这一结论。Run10 所有结果、权重、曲线和 bundle
必须原样保留。
