# Test2：LGLA v1 冻结测试协议

`TEST2_PROTOCOL_STATUS=PREREGISTERED`
`TRAINING_PERMITTED=FALSE`
`MODEL_SELECTION_PERMITTED=FALSE`
`TEST_EXECUTION_AUTHORIZED=FALSE`

## 目的与冻结边界

Test2 只评估 development 阶段已经锁定的 LGLA v1。模型、checkpoint、三 seed、输入尺寸、阈值、分类规则和指标均在读取 Test2 结果前固定；不训练、不微调、不校准、不改阈值，也不根据测试结果挑 seed。正式推理固定使用未融合的 fp32 EMA、CUDA autocast、八类 argmax、温度 1.0 和分割阈值 0.5。

三份冻结 checkpoint 为 LG2 seed42 visible epoch 74、LGX seed43 visible epoch 99、LG3 seed44 visible epoch 97。三 seed 全部报告均值和样本标准差；seed43 仅作为事先固定的展示代表，不获得模型选择特权。

## 数据与指标

- IMA++ test 共 634 张，全部用于 Binary IoU、Binary Dice、Pixel Accuracy。
- 其中 MEL 16 张、NV 40 张具有分类标签，只报告 Accuracy、MEL Recall、NV Recall、Balanced Recall、MEL→NV、NV→MEL、NLL、Brier、ECE；不得称为完整八类测试性能。
- ISIC2018 增量 external 共 89 张，只报告分割指标；它仍属于 ISIC 相关域，不称为独立医院验证。
- PAD-UFES-20 排除。Test2 不计算 development Joint，也没有事后接受门。
- 与已完成的 Plain Test1 比较时只做描述性对照，绝不覆盖或重跑 Test1。

## 一次性执行纪律

正式输出固定为 `experiments/test2/results`，存在即拒绝覆盖。失败 partial 原样保留审计。只有 `result.json` 是正式机器可读结果。用户于 2026-09-16 明确要求不足半小时的 test/val 由 Codex 直接运行；该请求作为本次 Test2 的新鲜授权记录，须在预检通过后写入发布门。
