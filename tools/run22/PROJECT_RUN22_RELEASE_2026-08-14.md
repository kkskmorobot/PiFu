# Run22 正式训练发布门

`PROJECT_RUN22_GATE=PASS`

`RUN22_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run22 的配置冻结、静态检查、单元/回归测试、CPU 与 GPU/AMP 断点恢复、YOLO26m 容量、prepare-only 和正式 Trainer 构造预检均已 PASS。

- 冻结配置 SHA-256：`a4ab4cf55726bed2b43e9cf6716be90523d8a087bd2fffd86ac89453abd7d25a`
- 就绪指纹：`6a9cc96bd2d21f28b9c17ed745a50e76846f05d9c96153b2bad69f59c7257e82`
- 正式入口：`tools/run22/train_run22_from_pycharm.py`
- 用户已于 2026-08-14 明确确认启动 Run22 正式训练。
- 正式结果目录尚未创建；本次只完成发布解锁，正式训练尚未启动。

Run22 发布条件已全部满足，可从 PyCharm 启动正式入口。仍不得运行 test 或外部候选评价。
