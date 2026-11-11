# Run21 正式训练发布门

`PROJECT_RUN21_GATE=PASS`

`RUN21_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run21 的配置冻结、静态检查、单元/回归测试、CPU 与 GPU/AMP 断点恢复、YOLO26m 容量、prepare-only 和正式 Trainer 构造预检均已 PASS。

- 冻结配置 SHA-256：`af4fa31a4cb9db0da5d090f70314d08f71667a05a078ee61af37dede9415abbb`
- 就绪指纹：`b7b853c4284037a6d51d0e66e10cd0302d167bf4ab30c7b32f512f6f137fb1a9`
- 正式入口：`tools/run21/train_run21_from_pycharm.py`
- 用户已于 2026-08-13 明确确认启动 Run21 正式训练。
- 正式结果目录尚未创建；本次只完成发布解锁，正式训练尚未启动。

Run21 发布条件已全部满足，可从 PyCharm 启动正式入口。仍不得运行 test 或外部候选评价。
