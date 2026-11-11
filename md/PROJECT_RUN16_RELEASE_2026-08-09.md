# PiFuV2 Run16 训练前发布门禁

日期：2026-08-09

`PROJECT_RUN16_GATE=PASS`  
`RUN16_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

发布对象：

- `experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention`；
- PyCharm 入口 `train_run16_from_pycharm.py`；
- 单一训练因素：Run12 的 `yolo26s-sem.pt` 替换为 `yolo26m-sem.pt`；
- Run12 BLG-MDA + unweighted CE 原样保留，不继承 Run13/14/15 loss；
- Run15 utility floors 仅修正 val alias 资格，early-stopping fitness 不变。

43/43 回归、实际权重真实 development batch forward/loss、CPU 完整恢复、RTX 5080
AMP workers=2 完整恢复、正式 Trainer 只构造和 16GB 容量检查均 PASS。冻结 behavior
fingerprint：
`2433474d4f826d8fcfed94e946eb9facf54d4ac13c20fdc5e2720855ddf18e27`。

用户已于 2026-08-09 在看到完整 Run15 结论、Run16 单因素方案和工程验收后明确回复：
`确认启动 Run16 正式训练`。确认门禁现已满足，可由用户从 PyCharm 无参数运行冻结
入口 `train_run16_from_pycharm.py`。本 release 只授权 Run16 正式 train/val 训练，
不授权 test 或 external evaluation。
