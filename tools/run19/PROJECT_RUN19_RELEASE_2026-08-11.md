# Run19 正式训练发布门

`PROJECT_RUN19_GATE=PASS`

`RUN19_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 当前判断

Run19 的配置锁、注意力继承与早停单元测试、CPU/GPU 完整中断恢复、本地 `tools/yolo26n.pt` AMP 自检、batch=16 容量、真实开发 batch 前向、正式 Trainer 构造和隔离回归均已通过。正式训练结果目录尚未创建。

- 配置 SHA-256：`03a0f9e2be6163cb9c51e06b3b4795e1f98a07e7e82755fd81eab61547051fb0`；
- ready 指纹：`e9bccce24376124422f226a4e7779faaaa8a9cea177c401e9a23b392fd76262c`；
- 正式模型：`models/yolo26m-sem.pt`；
- AMP 自检：`tools/yolo26n.pt`；
- 单一因素：相对 Run18 增加冻结的 Run16 BLG-MDA 注意力；
- batch=16、imgsz=640、epochs=100、epoch 40 后 patience=30；
- test 与外部候选评价禁止。

## 用户授权

2026-08-11，用户明确回复“确认启动 Run19 正式训练”。发布门现已放行；正式训练只能从 PyCharm 启动唯一入口 `tools/run19/train_run19_from_pycharm.py`。
