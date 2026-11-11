# Test1 发布门

`PROJECT_TEST1_GATE=PASS`
`TEST1_PREFLIGHT_GATE=PASS`
`TEST1_READY_GATE=CLOSED_COMPLETE`
`USER_CONFIRMATION_RECORDED=TRUE`
`ONE_TIME_FINAL_EVALUATION_AUTHORIZED=CONSUMED`
`TRAINING_PERFORMED=FALSE`
`TEST_METRICS_USED=TRUE`
`EXTERNAL_METRICS_USED=TRUE`
`PAD_UFES_20_USED=FALSE`

`ATTEMPT1_STATUS=INCOMPLETE_DATASET_INITIALIZATION_ERROR`
`ATTEMPT2_STATUS=INCOMPLETE_EXTERNAL_CLASSIFICATION_STATS_ERROR`
`ATTEMPT3_STATUS=INCOMPLETE_EXTERNAL_VALIDATOR_IMPORT_ERROR`
`TEST1_RESULT_STATUS=COMPLETE`
`POST_TEST_DEVELOPMENT_CHANGES_PERMITTED=FALSE`

## 当前结论

协议与入口已经建立，静态/指标契约检查和零张量 GPU 预检均已通过。用户于 `2026-09-11` 在看到准备结果后明确回复“确认执行 test1 一次性最终评估”，因此本次一次性正式执行门已开放。准备阶段没有读取 test/external 图像进行模型推理。

- 恢复预检时间：`2026-09-11 14:28:25 +0800`
- 配置 SHA-256：`c52579693178474e0bea80c8b5878a0420c7d6261f292bc3ad85900323801722`
- 锁定 fingerprint：`e6b1cfa834cc682922527cd1898ea6ea65a8f4999593bb70bba5e85c3fb20d42`

第一次启动在数据集初始化阶段因 `masks_dir` 映射错误中止；第二次在 Run18 external 分割推理结束后，因共享验证器错误地要求无分类标签的 external 计算分类统计而中止。两个失败 partial 均已原样归档。两次修复只涉及评估封装：正确映射 mask，并让 external 只返回预注册的分割指标；冻结 manifest、样本、checkpoint、模型、指标定义与决策规则均未改变。修复后的静态检查和零张量 GPU 预检重新通过，本次仍属于用户已经明确授权的同一次 Test1 操作恢复。

`preflight.json` 与 `ready.json` 已通过、锁定 fingerprint 未变化，并已获得用户新的明确确认，因此以下三项已同时设为：

- `TEST1_READY_GATE=PASS`
- `USER_CONFIRMATION_RECORDED=TRUE`
- `ONE_TIME_FINAL_EVALUATION_AUTHORIZED=TRUE`

任何单独修改、口头推断或历史授权均不足以开放正式入口。正式结果目录一旦存在即禁止覆盖；失败产生的 partial 目录必须先审计，不能静默绕过。

## 完成记录

- 完成时间：`2026-09-11 14:31:05 +0800`
- 正式结果：`experiments/test1/results/result.json`
- 结果大小：`12901` bytes
- 结果 SHA-256：`5825323ce93a21272b37ae4b0a84d819e48d9edcff6080ed580e942461dd5557`
- 正式 fingerprint：`e6b1cfa834cc682922527cd1898ea6ea65a8f4999593bb70bba5e85c3fb20d42`
- 完成 seed：`42,43,44`

一次性授权已经消费并关闭。结果好坏均不得用于返回 development 修改模型、权重、阈值、决策规则或重新选择 seed。
