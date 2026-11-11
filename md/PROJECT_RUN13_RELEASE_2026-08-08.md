# PiFuV2 Run13 训练前发布门禁

日期：2026-08-08

`PROJECT_RUN13_GATE=PASS`  
`RUN13_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

## 发布对象

- 实验：`experiments/13_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELWeightedCE`；
- PyCharm 唯一正式入口：`train_run13_from_pycharm.py`；
- 预注册配置：`run13_preregistered_config_v1.yaml`；
- 单一变化：在 Run12 BLG-MDA 上加入 Run11 冻结 MEL target CE 权重。

## 当前状态

预注册、25 项回归测试、CPU/GPU 完整恢复、正式 Trainer 构造、真实 development
batch forward/loss 与冻结 fingerprint 全部 PASS。behavior fingerprint 为
`980198023b0119e63882eb5d5f5fedf662691eb4837a775c5b73b0696d38375b`。

正式结果目录不存在，活动 Run13 训练进程为 0；optimizer、backward、optimizer step
和 `trainer.train()` 均未在最终 prepare-only 中调用。

用户已于 2026-08-09 明确确认启动 Run13 正式训练，因此记录为
`USER_CONFIRMATION_RECORDED=TRUE`。该确认只授权使用冻结的 PyCharm 入口启动 Run13，
不授权修改预注册因素，也不授权 test 或 external evaluation。
