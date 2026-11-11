# PiFuV2 Run14 训练前发布门禁

日期：2026-08-09

`PROJECT_RUN14_GATE=PASS`  
`RUN14_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=TRUE_INTERRUPTED_PRE_EPOCH1`  
`RUN14_POWER_LOSS_RECOVERY_GATE=PASS`  
`CURRENT_RESUME_MODE=FRESH_START_NO_COMPLETE_BUNDLE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

发布对象：

- `experiments/14_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVLogitMargin`；
- PyCharm 入口 `train_run14_from_pycharm.py`；
- 唯一因素 TMNCM，margin=`ln(5539/3047)=0.5976564790473948`。

工程验收已完成：32/32 相关回归测试、CPU 完整恢复、RTX 5080 AMP workers=2
完整恢复、真实 development batch 前向/损失与正式 Trainer 只构造检查均 PASS。
冻结 behavior fingerprint：
`30e88fffb1eab5c59d7a3a1e0cbb458dcd2b020414c1011853941ce9fd456bc4`。

用户已于 2026-08-09 在看到本次准备结果后明确回复：
`确认启动 Run14 正式训练`。确认门禁现已满足，可由用户从 PyCharm 运行冻结入口
`train_run14_from_pycharm.py`。本 release 只授权 Run14 正式 train/val 训练，不授权
test 或 external evaluation。

首次正式启动在第 1 个 epoch 完成前因电脑意外断电终止，没有产生完整 `_SUCCESS`
bundle。断电把 Run14 协议和 development manifest 截成 0 字节，故入口按设计 fail
closed。两个 0 字节文件和半成品结果目录已完整归档；冻结原件已按训练前 SHA-256
恢复，全部门禁和 fingerprint 复核 PASS。当前正式结果目录不存在，再次无参数运行
同一入口应 fresh start，不得尝试恢复半个 epoch。
