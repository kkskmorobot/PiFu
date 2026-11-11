# Run23 正式训练发布门

`PROJECT_RUN23_GATE=PASS`

`RUN23_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run23 已按 Run22 结果预注册为 P3/P4 ECA-only seed43 配对重复。配置冻结、静态测试、CPU 与 GPU/AMP 完整状态恢复、YOLO26m batch=16 容量步、`--prepare-only` 和正式 Trainer 只构造预检均已 PASS；正式实验目录仍不存在。

- 配置 SHA-256：`6e85f7544c034960ffaf68e8a9268b15367c311cb484c301ed52f19f698ebf44`
- ready fingerprint：`8837f6958bf63820732ad852367dd2d638927fcdd9de0d0a96c039c2cdf55c13`
- 正式入口：`tools/run23/train_run23_from_pycharm.py`

用户已于 2026-08-15 明确回复“确认启动 Run23 正式训练”。正式训练尚未由本次发布门写入操作启动，正式实验目录仍应在 PyCharm 实际运行入口后才创建。

正式入口：`tools/run23/train_run23_from_pycharm.py`。

Run23 发布条件已经全部满足，可以从 PyCharm 运行正式入口。仍不得运行 test 或外部候选评价。
