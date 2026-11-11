# Run20 正式训练发布门

`PROJECT_RUN20_GATE=PASS`

`RUN20_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run20 的静态检查、单元/回归测试、CPU 与 GPU/AMP 断点恢复、YOLO26m 容量、prepare-only 和正式 Trainer 构造预检均已 PASS。

- 冻结配置 SHA-256：`65bd40b9241c218b1ff4f403990a99390ad47b02874827801479a03458417338`
- 就绪指纹：`38db3f494705668e995328c1db4491f00f32d49d1ce1149a9df8cc677ff10d1e`
- 正式入口：`tools/run20/train_run20_from_pycharm.py`
- 用户已于 2026-08-12 明确确认启动 Run20 正式训练。
- 正式结果目录尚未创建；本次只完成发布解锁，正式训练尚未启动。

Run20 发布条件已全部满足，可从 PyCharm 启动正式入口。仍不得运行 test 或外部候选评价。
