# Run21 预检复核

`RUN21_PREFLIGHT_VERIFICATION=PASS`

`FORMAL_TRAINING_STARTED=FALSE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 冻结对象

- 配置：`run21_preregistered_config_v1.yaml`
- 配置 SHA-256：`af4fa31a4cb9db0da5d090f70314d08f71667a05a078ee61af37dede9415abbb`
- prepare-only 就绪指纹：`b7b853c4284037a6d51d0e66e10cd0302d167bf4ab30c7b32f512f6f137fb1a9`
- 预训练模型：`models/yolo26m-sem.pt`
- 模型 SHA-256：`3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`
- AMP 检查资产：`tools/yolo26n.pt`
- AMP 资产 SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

## 已通过项目

1. `py_compile` 静态语法检查：PASS。
2. Run21 新增单元测试：4 passed。
3. 项目级受控回归：66 passed、2 subtests passed。
4. CPU 合成 train/val 端到端中断恢复：PASS；连续与恢复后的模型、EMA、优化器、随机状态和 batch 顺序逐项相等。
5. RTX 5080 GPU/AMP 合成 train/val 中断恢复：PASS；全部比较项最大绝对差为 0。
6. YOLO26m、640、batch 16、AMP 合成前后向及 AdamW step：PASS；峰值 reserved 6,996,099,072 bytes，占设备显存 40.93%。
7. prepare-only 数据/配置/模型锁与 GPU 前向：PASS；`trainer.train()` 未调用。
8. 正式 Trainer 构造：PASS；模型为 `P3P4ECAMultiTaskModel`，训练器为 `DelayedUtilityGatedP3P4ECATrainer`，正式结果目录未创建。

## Run21 唯一预注册改动

相对 Run18，只在原 P3/P4 分类特征上各增加一个 kernel size 5、全零卷积初始化的 ECA 门，再执行不变的 global-average pooling。共新增 10 个可学习参数；不使用空间病灶 mask，不添加 P5，语义分割分支不修改。其余数据、训练、统一 val、checkpoint 和延迟早停协议均锁定不变。

## 当前门禁

`RUN21_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN21_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

用户已于 2026-08-13 明确回复“确认启动 Run21 正式训练”。发布门已改为 PASS，可从 PyCharm 运行 `tools/run21/train_run21_from_pycharm.py`。本次未从命令行代为启动训练，正式结果目录应在入口实际运行后才创建。
