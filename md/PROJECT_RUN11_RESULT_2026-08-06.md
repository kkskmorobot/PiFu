# PiFuV2 Run11 正式结果登记

日期：2026-08-06  
实验：`experiments/11_YOLO26s_Binary8Class_MELWeightedCE`

`RUN11_TRAINING_EXECUTION_GATE=PASS_NORMAL_EARLY_STOP`  
`RUN11_CHECKPOINT_INTEGRITY_GATE=PASS`  
`RUN11_FEASIBILITY_GATE=FAIL_NO_FEASIBLE_CHECKPOINT`  
`RUN11_FORMAL_MODEL_LOCK=NOT_CREATED`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 已核实事实

- 恢复后日志含连续 epoch 1~83 共 83 行，非有限值为 0；
- 完成标记 `last_completed_epoch=82`；
- 5 个保留 bundle 通过大小、SHA-256、fingerprint 和必需状态验证，auto resume
  指向 `epoch_0082`；
- PyCharm 入口 SHA-256 恢复为预注册值
  `ee1ad4c0d4d5048af701d090bbb6f9fa880ba2ce7899b297641fb5ba9e86c148`；
- 没有异常/安全中断标记，没有使用 test 或 external evaluator。

## val-only 结果

- 最佳 binary IoU / Dice：日志 epoch 20，0.77894 / 0.87574；
- 最佳 group Macro-F1 / joint：日志 epoch 63，0.57017 / 0.66254；
- 最佳 image Macro-F1：日志 epoch 72，0.51780；
- 全程最低 MEL->NV：日志 epoch 4，0.11224；
- 冻结上限为 0.1077844311，83 个 epoch 的 feasible 数量为 0；
- `best_feasible_joint` alias 不存在。

## 与 Run10 的受控结论

MEL target CE 权重 1.8178536265 使最低 MEL->NV 从 0.32653 降至 0.11224，
绝对下降 0.21429；但最佳 image Macro-F1、group Macro-F1 和 joint 分别下降
0.02513、0.02409 和 0.01878。该因素明显改善风险方向，但未越过约束并牺牲整体
分类/联合性能。

Run11 登记为约束失败诊断实验，所有结果和 bundle 原样保留；不得以
`best_joint_score` 冒充正式锁定模型，也不得借助 test/external 改变结论。
