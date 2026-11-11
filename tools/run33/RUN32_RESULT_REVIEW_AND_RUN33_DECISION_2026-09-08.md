# Run32 结果复核与 Run33 决策（2026-09-08）

`RUN32_RESULT_REVIEW=COMPLETE`

`NEXT_EXPERIMENT_DECISION=RUN33_YOLO26M_P3P4_STANDARD_CBAM_SEED43`

## Run32 完整性

- 完整训练至可见 epoch 100，CSV epoch 1～100 连续，无非有限值；`last_completed_epoch=99`。
- 最佳 Joint bundle 为零基 `epoch_0080`（可见 epoch 81）。
- 5 个完整 bundle（`0043/0080/0097/0098/0099`）共 10 个权重/状态文件；实际大小与
  SHA-256 均和 manifest 一致，`_SUCCESS`、epoch 和 fingerprint 全部通过。
- fingerprint 为 `df95baaec5fde165c600f8ace75cc15c0ca4ef4076aff8f899d51b050da1862c`。
- 未使用 IMA++ test、ISIC2018 external 或 PAD-UFES-20。

## 最佳 Joint 与同 seed43 Run22 配对

| 指标 | Run22 | Run32 | Run32 - Run22 |
|---|---:|---:|---:|
| Binary IoU | 0.76377 | 0.77082 | +0.00705 |
| Image Macro-F1 | 0.55017 | 0.52794 | -0.02223 |
| Group Macro-F1 | 0.62811 | 0.57821 | -0.04990 |
| Group Recall | 0.64033 | 0.57467 | -0.06566 |
| MEL→NV | 0.40816 | 0.41837 | +0.01021（恶化） |
| malignant→benign | 0.28502 | 0.30435 | +0.01933（恶化） |
| NLL | 0.78156 | 0.84011 | +0.05855 |
| Brier | 0.36710 | 0.36606 | -0.00104 |
| ECE | 0.07998 | 0.09126 | +0.01128 |
| Joint | 0.69263 | 0.66760 | -0.02503 |

Run32 有 77 个 utility-eligible epoch，但正式 feasible 与 risk-feasible checkpoint 均为 0。

## 判定与下一步

Run32 不满足“两个风险指标实质改善且 Group Macro-F1/Joint 无明显损害”的预注册门。恶性/良性
训练期辅助头不仅没有降低风险，反而同时损害分类、group 效用和 Joint。因此：

1. 不做 seed42/44 复现；
2. 连同 Run29～Run31，关闭训练期风险损失/辅助头探索路线；
3. 不根据 Run32 validation 临时修改辅助权重、类别定义或隐藏层；
4. 按根目录 `PROJECT_POST_RUN32_EXPERIMENT_PLAN_2026-09-08.md`，下一步补做 YOLO26m
   P3/P4 标准 CBAM seed43 单因素对照；
5. Run33 只与 Run22 配对，保持 partial-label、数据、损失、训练生命周期、指标和选择协议不变。

