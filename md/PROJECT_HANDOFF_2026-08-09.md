# PiFuV2 当前交接（Run16 已确认、点击前）

日期：2026-08-09  
本文件是当前权威状态增量；此前 handoff、Run1~Run15 结果与 checkpoint 原样保留。

## Run15 结论

- Run15 可见 epoch 58 正常早停，58 行连续、非有限值 0；
- 5 个完整 bundle 全部通过验证，auto resume 指向内部 epoch 57；
- 最佳 binary IoU/Dice=0.77587/0.87379，最佳 group Macro-F1/joint=
  0.57479/0.66724，均低于 Run12 的 0.59969/0.68258；
- 成熟最佳 joint 状态 MEL->NV=0.29592；唯一低风险点是冷启动退化状态；
- 非退化 feasible 数量 0，未建立模型锁；不续训，不继承 pairwise loss。

## Run16 选择与唯一因素

Run16 回到 Run12 的 BLG-MDA + unweighted CE，只把官方预训练骨干从 YOLO26s-sem
升级为 YOLO26m-sem。权重由用户手动下载，SHA-256 为
`3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`。

数据、manifest/split、mask、partial-label scope、batch=16、640、epochs=100、patience=20、
AdamW、LR、增强、AMP、seed=42 全部不变。Run15 utility floors 只用于排除冷启动 alias，
不改 fitness 或训练轨迹。

## 工程验证与恢复

- `py_compile`、43/43 tests、manifest 字节一致性 PASS；
- actual pretrained transfer 420/431，真实 development batch forward/loss PASS；
- CPU 和 RTX 5080 AMP workers=2 的完整中断恢复逐状态最大差值 0；
- 正式 Trainer 只构造 PASS，optimizer 未创建，train 未调用；
- full-step 容量预检 reserved 约 6.51 GiB，16GB PASS；
- behavior fingerprint：
  `2433474d4f826d8fcfed94e946eb9facf54d4ac13c20fdc5e2720855ddf18e27`；
- 正式 Run16 结果目录不存在；test/external 均未评价。

## 当前门禁与下一动作

`PROJECT_RUN16_GATE=PASS`  
`RUN16_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

正式入口已经冻结为：
`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run16_from_pycharm.py`。

用户已于 2026-08-09 明确回复 `确认启动 Run16 正式训练`。下一动作只有一个：由用户
从 PyCharm 无参数运行上述入口，默认 `resume=auto`。当前正式结果目录不存在，首次运行
将 fresh start；以后如安全中断，再次运行同一入口会恢复最新完整 bundle。正式模型锁
前禁止 test；ISIC2018 Task 1 与 PAD-UFES-20 继续仅作外部候选。
