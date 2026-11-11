# Run16 正式训练启动授权记录

日期：2026-08-09  
实验：`experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention`

`RUN16_START_AUTHORIZATION=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

用户在看到 Run15 正式结果、Run16 的 YOLO26m 单因素方案、权重校验、43/43 回归、
真实 development batch 前向、CPU/GPU 完整恢复和最终点击前门禁后，明确回复：

`确认启动 Run16 正式训练`

本授权只允许用户从 PyCharm 无参数运行项目根目录
`train_run16_from_pycharm.py`，执行冻结的 Run16 train/val。它不授权运行 test、
ISIC2018 external 或 PAD-UFES-20 external。首次启动默认 fresh start；如以后在完整
epoch 边界安全中断，再次运行同一入口使用默认 `resume=auto` 恢复最新完整 bundle。
