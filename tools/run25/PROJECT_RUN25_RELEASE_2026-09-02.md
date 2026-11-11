# Run25 正式训练发布门

`PROJECT_RUN25_GATE=PASS`

`RUN25_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run25 已预注册为 Run24 的 P3/P4 ECA-only seed44 严格配对重复。配置冻结、7 项静态测试、CPU 与 RTX 5080/AMP 完整状态恢复、YOLO26m batch=16 容量步、`--prepare-only` 和正式 Trainer 只构造预检均已 PASS。配置 SHA-256 为 `05cec4a8dde18e93858d18b352b8429edc4ed592f14db65c56f680ca5969927b`，最终 fingerprint 为 `c2cf59e05a941c81d744bc47e722cebcc61bd24da715cfe48e34970048dd6910`，正式实验目录仍不存在。

用户已于 2026-09-02 明确回复“确认启动 Run25 正式训练”。Run25 发布条件现已全部满足，可以从 PyCharm 无参数运行 `tools/run25/train_run25_from_pycharm.py`。本次仅解锁发布门，尚未启动正式训练。

正式训练前及训练过程中仍禁止运行 test 或外部候选评价。
