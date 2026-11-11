# PiFuV2 Run15 训练前发布门禁

日期：2026-08-09

`PROJECT_RUN15_GATE=PASS`  
`RUN15_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

发布对象：

- `experiments/15_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVPairwiseBoundary`；
- PyCharm 入口 `train_run15_from_pycharm.py`；
- 唯一训练因素 DPMNB：Run12 unweighted CE 加 true-MEL direct pairwise softplus；
- metric protocol 修正仅阻止冷启动退化 alias，不改变 early-stopping fitness。

工程验收已完成：40 项回归与 2 个 subtests、CPU 完整恢复、RTX 5080 AMP workers=2
完整恢复、真实 development batch 前向/损失与正式 Trainer 只构造检查均 PASS。
冻结 behavior fingerprint：
`f2e38bd9fd0422280de8c1e8ac3520d5bf6f0843efaef343d7e0601f3a43b978`。

用户已于 2026-08-09 在看到本次结果后明确回复：
`确认启动 Run15 正式训练`。确认门禁现已满足，可由用户从 PyCharm 无参数运行冻结
入口 `train_run15_from_pycharm.py`。本 release 只授权 Run15 正式 train/val 训练，
不授权 test 或 external evaluation。
