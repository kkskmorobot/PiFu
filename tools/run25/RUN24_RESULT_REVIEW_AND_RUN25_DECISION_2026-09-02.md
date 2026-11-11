# Run24 结果复核与 Run25 决策

`RUN24_RESULT_REVIEW=PASS`

`NEXT_EXPERIMENT_DECISION=RUN25_P3P4_ECA_ONLY_SEED44_REPLICATION`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## Run24 完成状态

- 正常完成 100 个 CSV epoch（1~100），Trainer 最后完成 epoch 为 99；存在 `RUN24_TRAINING_COMPLETE.json`。
- 最佳联合选择别名为 checkpoint bundle `epoch_0077`，对应 `results.csv` epoch 78。
- 该点 Binary IoU=0.76393、Image Macro-F1=0.54301、Group Macro-F1=0.60641、Group Macro-Recall=0.62047、mel→nv rate=0.34694、Joint=0.68063。
- 100 个 epoch 中没有满足 mel→nv≤0.1077844311 的风险可行点；因此 Run24 不产生可作为正式候选的 `best_feasible_joint`，最佳 joint 只作诊断比较。
- 全程未读取 test，也未使用外部候选集选择模型或阈值。

## 已完成的同 seed 配对

checkpoint bundle epoch 为 0 基，表中指标使用与之对应的 `results.csv epoch=bundle_epoch+1`。

| seed | plain | ECA | Δ Binary IoU | Δ Image Macro-F1 | Δ Group Macro-F1 | Δ mel→nv | Δ Joint |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | Run18 | Run21 | +0.01108 | -0.00672 | -0.00804 | +0.02040 | +0.00039 |
| 43 | Run22 | Run23 | -0.00196 | +0.00028 | -0.04049 | +0.00000 | -0.02356 |

ECA 的 joint 差值在两个 seed 间方向和幅度不稳定，且 Group Macro-F1 在两组中均下降。现有证据不足以支持或否定注意力模块的稳定净效应。

## Run25 决策

按既定消融顺序补齐第三组 seed44 配对：Run25 严格复制 Run23 的 P3/P4 ECA-only 结构，并与 Run24 使用相同 seed=44。相对 Run24 唯一预注册训练因素是给分类头 P3、P4 特征各增加一个 kernel size 5、identity-initialized 的 ECA 通道门。

数据、预训练权重、batch、增强、损失、采样、优化器、100 epoch 预算、延迟早停、风险阈值、统一 val 指标和 checkpoint 合同全部保持不变。Run25 完成后再对三个 seed 的配对差值做汇总；此前不增加新模块，不改阈值，不运行 test 或外部评价。
