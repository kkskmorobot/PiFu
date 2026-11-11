# Run20 预检复核

`RUN20_PREFLIGHT_VERIFICATION=PASS`

`FORMAL_TRAINING_STARTED=FALSE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 冻结对象

- 配置：`run20_preregistered_config_v1.yaml`
- 配置 SHA-256：`65bd40b9241c218b1ff4f403990a99390ad47b02874827801479a03458417338`
- prepare-only 就绪指纹：`38db3f494705668e995328c1db4491f00f32d49d1ce1149a9df8cc677ff10d1e`
- 预训练模型：`models/yolo26m-sem.pt`
- 模型 SHA-256：`3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`
- AMP 检查资产：`tools/yolo26n.pt`
- AMP 资产 SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

## 已通过项目

1. `py_compile` 静态语法检查：PASS。
2. Run20 新模块单元测试：4 passed。
3. 项目级受控回归：62 passed、2 subtests passed。
4. CPU 合成 train/val 端到端中断恢复：PASS；连续与恢复后的模型、EMA、优化器、随机状态和 batch 顺序逐项相等。
5. RTX 5080 GPU/AMP 合成 train/val 中断恢复：PASS；全部比较项最大绝对差为 0。
6. YOLO26m、640、batch 16、AMP 合成前后向及 AdamW step：PASS；峰值 reserved 6,994,001,920 bytes，占设备显存 40.91%。
7. prepare-only 数据/配置/模型锁与 GPU 前向：PASS；`trainer.train()` 未调用。
8. 正式 Trainer 构造：PASS；模型为 `AdaptiveResidualBinaryLesionGuidedAttentionModel`，训练器为 `DelayedUtilityGatedAdaptiveResidualAttentionTrainer`，正式结果目录未创建。

## Run20 唯一预注册改动

Run19 的 P3/P4/P5 固定共享 `mask_guided_mix=0.5`，替换为三个逐尺度、可学习、上限 0.5 的残差门；三个门均从 0.05 初始化。数据、split、manifest、损失、batch、分辨率、优化器、seed、统一 val、joint-score-2、checkpoint 和延迟早停协议不变。

## 当前门禁

`RUN20_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN20_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

用户已于 2026-08-12 明确回复“确认启动 Run20 正式训练”。发布门已改为 PASS，可从 PyCharm 运行 `tools/run20/train_run20_from_pycharm.py`。本次未代为启动训练，正式结果目录仍应在入口实际运行后才创建。
