# Test1 一次性最终评估协议

`TEST1_PROTOCOL_STATUS=PREREGISTERED`
`TRAINING_PERMITTED=FALSE`
`MODEL_SELECTION_PERMITTED=FALSE`
`TEST_EXECUTION_AUTHORIZED=FALSE`

## 目的

Test1 是 development 阶段关闭后的唯一正式最终评估。它不训练、不微调、不校准、不修改阈值，也不根据测试结果挑选 seed。最终架构固定为 Plain YOLO26m P3/P4 partial-label 多任务模型，推理规则固定为温度 `1.0` 的八类 argmax，输入尺寸固定为 `640`，二值语义阈值固定为 `0.5`。

## 冻结模型

一次执行必须依次评估全部三个已锁定 best-Joint bundle：Run18 seed42 epoch 50、Run22 seed43 epoch 60、Run24 seed44 epoch 78。正式报告给出每个 seed 及三 seed 均值和样本标准差。不得根据 Test1 结果只保留表现最好的 seed。Run22 seed43 是在查看 Test1 前已经指定的单模型代表。

模型结构从各 bundle 的 `trainer.pt` 恢复，权重从同一 bundle 的 `runtime_state.pt::training_state.ema` 无损 fp32 状态恢复。正式推理沿用 Val1 已验证的训练期数值路径：未融合 Conv+BN、fp32 EMA、CUDA autocast、无梯度。不得以压缩存储的 fp16 EMA 替代。

## 数据角色

### IMA++ 内部 test

- 冻结样本数：634；全部具有二值分割真值。
- 其中只有 56 张具有官方分类标签，且仅覆盖 MEL 16 张与 NV 40 张；这 56 张分别属于 56 个 group。
- 分割指标在全部 634 张上计算。
- 分类结果只称为“IMA++ test 的 MEL/NV 标签子集结果”，不得称为完整八类测试性能。
- 禁止计算或报告八类 Macro-F1、八类 Joint 或正式 feasible checkpoint 数，因为该子集缺少另外六类支持。

### ISIC2018 external

- 只使用与 IMA++ 经过 ID、SHA-256 和严格感知签名去重后保留的 89 张 Task-1 增量样本。
- 只报告二值分割，不进行疾病分类。
- 该集合仍属于 ISIC 相关域，不得描述为独立医院或真正独立临床外部验证。

### 排除项

PAD-UFES-20 不进入 Test1。ISIC2019 官方 challenge test、leaderboard 结果及任何其他外部来源均不进入 Test1。

## 预注册指标

每个 seed 分别报告：

1. IMA++ test 全部 634 张的 Binary IoU、Binary Dice、pixel accuracy；
2. 56 张 MEL/NV 标签子集的八类 argmax accuracy、MEL recall、NV recall、balanced recall、MEL→NV、NV→MEL、NLL、八类概率 Brier 与 ECE；
3. 上述六项离散分类指标的固定 seed `20260911`、按真实 MEL/NV 类别分层的 10,000 次 group bootstrap 95% percentile CI；
4. ISIC2018 增量 89 张的 Binary IoU、Binary Dice、pixel accuracy；
5. 三 seed 的均值与样本标准差。

分类 argmax 允许预测到八类中的任意类别。MEL recall 只在真实 MEL 上计算，NV recall 只在真实 NV 上计算；balanced recall 为二者平均。MEL→NV 与 NV→MEL 只统计这两个特定方向，不能把预测到其他类别的错误隐藏进去。

## 最终评估纪律

- Test1 没有接受门，是描述性最终评估；结果好坏都必须保留并报告。
- 完成后不得返回 development 阶段修改模型、权重、决策规则、输入尺寸、taxonomy 或数据划分。
- 失败的 partial 输出保留并标记不完整；只有原子发布的 `result.json` 才是正式结果。
- 正式执行前必须有 `preflight.json`、`ready.json`、发布门 PASS，以及用户在看到本协议与门禁后给出的新的明确确认。
- 用户已授权不足半小时的验证/测试由 Codex 后台执行；这一授权只改变执行方式，不替代本次一次性数据访问的最终确认门。

## 当前状态

当前只准备协议与发布门，不运行 IMA++ test 或 ISIC2018 external。`release.md` 保持等待用户确认，正式入口必须在该门改变前拒绝执行。
