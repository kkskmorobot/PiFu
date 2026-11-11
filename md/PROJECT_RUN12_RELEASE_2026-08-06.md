# PiFuV2 Run12 训练前发布门禁

日期：2026-08-06

`PROJECT_RUN12_GATE=PASS`  
`RUN12_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

## 发布对象

- 实验：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`；
- PyCharm 唯一正式入口：`train_run12_from_pycharm.py`；
- 预注册配置：`run12_preregistered_config_v1.yaml`；
- behavior fingerprint：`77ba2344a10784443d11ded2a9e6dff497e92429481a986a3539becdbc45710f`。

## 当前状态

Run12 的协议、实现、19 项自动测试、CPU/GPU 完整状态断点恢复、正式 Trainer 构造和真实 development batch GPU forward/loss 均已通过。没有创建 Run12 正式结果目录，没有启动 optimizer step，没有运行 test 或 external evaluator。

用户已于 2026-08-06 明确回复“确认启动 Run12 正式训练”，因此记录为 `USER_CONFIRMATION_RECORDED=TRUE`。该确认只授权使用预注册入口启动 Run12 正式训练，不授权修改冻结因素，也不授权 test 或 external evaluation。

本发布门禁不授权 test。模型正式选择完成后，仍需最新文档明确授权并由用户再次确认，才可运行 test；PAD-UFES-20 与 ISIC 2018 Task 1 增量集继续只作外部候选。
