# Run24 正式训练发布门

`PROJECT_RUN24_GATE=PASS`

`RUN24_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run24 已预注册为 plain seed44 第三配对基线。配置冻结、静态测试、CPU 与 GPU/AMP 完整恢复、batch=16 容量步、`--prepare-only` 和正式 Trainer 只构造预检均已 PASS；正式实验目录仍不存在。

- 配置 SHA-256：`96d87780853c73b21ca5171028aeff2e679133f03e972cfdce0ade7b4122a3f9`
- ready fingerprint：`c856d97b12202fac2e400ecbcd94d5ad91c705981d6c692cda827e99e28929e6`
- 正式入口：`tools/run24/train_run24_from_pycharm.py`

正式入口：`tools/run24/train_run24_from_pycharm.py`。

用户已于 2026-08-16 明确回复“确认启动 Run24 正式训练”。Run24 发布条件已经全部满足，可以从 PyCharm 运行正式入口。本次只解锁发布门，尚未启动训练；仍不得运行 test 或外部候选评价。
