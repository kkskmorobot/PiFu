# Run22 预检复核

`RUN22_PREFLIGHT_VERIFICATION=PASS`

`FORMAL_TRAINING_STARTED=FALSE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 冻结对象

- 配置：`run22_preregistered_config_v1.yaml`
- 配置 SHA-256：`a4ab4cf55726bed2b43e9cf6716be90523d8a087bd2fffd86ac89453abd7d25a`
- prepare-only 就绪指纹：`6a9cc96bd2d21f28b9c17ed745a50e76846f05d9c96153b2bad69f59c7257e82`
- 预训练模型：`models/yolo26m-sem.pt`
- 模型 SHA-256：`3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`
- AMP 检查资产：`tools/yolo26n.pt`
- AMP 资产 SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

## 已通过项目

1. `py_compile` 静态语法检查：PASS。
2. Run22 seed43/早停契约测试：4 passed。
3. Run18–Run22 项目级回归：70 passed、2 subtests passed。
4. CPU seed43 合成 train/val 中断恢复 v2：PASS；连续与恢复后的模型、EMA、优化器、随机状态和 batch 顺序逐项相等。
5. RTX 5080 GPU/AMP seed43 中断恢复 v2：PASS；全部比较项最大绝对差为0。
6. YOLO26m、640、batch16、AMP合成前后向及AdamW step：PASS；峰值reserved 6,996,099,072 bytes，占设备显存40.93%。
7. prepare-only 数据/配置/模型锁与GPU前向：PASS；`trainer.train()`未调用。
8. 正式 Trainer v2构造：PASS；seed=43，模型为无注意力 `BinaryPartialMultiTaskModel`，正式结果目录未创建。
9. Run18/Run22同名模块冲突已通过Run22唯一模块名 `imaplusplus_partial_multitask_plain_seed43_v1.py` 消除；最终回归验证通过，Run18未修改。

## Run22 唯一预注册改动

相对Run18，只把正式随机种子从42改为43。plain P3/P4 head、无注意力、数据、损失、采样、增广、batch、分辨率、优化器、统一val、checkpoint和早停协议不变。Run22完成后，Run23将用同一个seed43重复Run21 ECA-only，形成严格配对。

## 当前门禁

`RUN22_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN22_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

用户已于2026-08-14明确回复“确认启动 Run22 正式训练”。发布门已改为PASS，可从PyCharm运行 `tools/run22/train_run22_from_pycharm.py`。本次未从命令行代为启动训练，正式结果目录应在入口实际运行后才创建。
