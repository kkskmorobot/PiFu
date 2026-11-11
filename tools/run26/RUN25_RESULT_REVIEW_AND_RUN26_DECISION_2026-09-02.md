# Run25 结果复核与 Run26 决策

`RUN25_RESULT_REVIEW=PASS`

`NEXT_EXPERIMENT_DECISION=RUN26_P3P4_ECA_IDENTITY_SPATIAL_ATTENTION_SEED44`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## Run25 完成状态

- `RUN25_TRAINING_COMPLETE.json` 存在；CSV 共 73 个 epoch，Trainer 最后完成 epoch=72。
- 风险惩罚后的早停适应度在 CSV epoch 43 最佳，之后连续 30 个 epoch 未改善，因此在 CSV epoch 73 正常早停；不是崩溃。
- 最佳联合选择为 checkpoint bundle `epoch_0047`，对应 CSV epoch 48。
- 该点 Binary IoU=0.76441、Image Macro-F1=0.52761、Group Macro-F1=0.62371、Group Macro-Recall=0.58258、mel→nv=0.44898、Joint=0.69049。
- 没有任何 epoch 满足 mel→nv≤0.1077844311，故最佳 joint 只能用于诊断，不能冒充风险可行候选。

## 三组 plain/ECA 严格配对结论

下表为 ECA−plain；checkpoint bundle epoch 已按 `results.csv epoch=bundle_epoch+1` 正确对齐。

| seed | Δ Binary IoU | Δ Image Macro-F1 | Δ Group Macro-F1 | Δ Group Recall | Δ mel→nv | Δ Joint |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.01108 | -0.00672 | -0.00804 | +0.00422 | +0.02040 | +0.00039 |
| 43 | -0.00196 | +0.00028 | -0.04049 | -0.07689 | +0.00000 | -0.02356 |
| 44 | +0.00048 | -0.01540 | +0.01730 | -0.03789 | +0.10204 | +0.00986 |
| 平均 | +0.00320 | -0.00728 | -0.01041 | -0.03685 | +0.04081 | -0.00444 |

ECA 对 Binary IoU 有很小的平均增益，但分类、group recall、风险和 joint 没有稳定收益，且随机种子间方向不一致。ECA-only 不通过“稳定注意力改进”结论门。

## Run26 决策

不再增加 ECA-only 重复。Run26 以 Run25 为基线，唯一新增因素是给 P3、P4 的 ECA 输出各增加一个 CBAM 风格空间门：沿通道计算 mean/max，拼接后用 7×7 卷积产生空间权重，采用 `2*sigmoid` 缩放。空间卷积全零初始化，使门初始严格等于 1，模型起点与 Run25 完全一致。

Run26 不使用预测掩码、不增加 P5、不修改语义分支，也不改变数据、loss、seed44、优化器、batch、训练预算、早停、风险阈值或 checkpoint 合同。若 Run26 相对 Run25 显示明确收益，再补 seed42/43；若仍无收益，则结束这一架构方向并转向独立预注册的风险损失实验。
