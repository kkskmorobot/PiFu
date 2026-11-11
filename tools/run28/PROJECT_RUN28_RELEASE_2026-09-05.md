# Run28 正式训练发布门

`PROJECT_RUN28_GATE=PASS`

`RUN28_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run28 已预注册为 Run23 seed43 P3/P4 ECA-only 的恒等初始化 spatial7 单因素配对复现。全部技术预检已通过。

- 配置 SHA-256：`ac9ab1df3e3e6a4d497ca0b08ae161860d5df8120ae7c4bce909d918ece5aca7`
- 最终就绪指纹：`5208445478cf276667f5f3f38e907bfc72fa9f93ac05fd38f794db4944ec0403`
- Run23 配对基线指纹：`8837f6958bf63820732ad852367dd2d638927fcdd9de0d0a96c039c2cdf55c13`
- 静态与回归测试：`8 passed`
- CPU 与 GPU/AMP/workers=2 断点恢复：连续与恢复状态逐项零差异
- RTX 5080 容量预检：batch=16、imgsz=640、AMP、AdamW 单步通过，峰值 reserved 显存占比约 41.47%
- development `--prepare-only` 与正式 Trainer 构造：通过，均未调用正式 `trainer.train()`
- 正式 Run28 实验目录：不存在

用户于 2026-09-05 要求根据 Run27 结果作出选择，并把选定的下一轮准备到由其本人点击训练之前；该确认已记录。允许用户从锁定入口 `tools/run28/train_run28_from_pycharm.py` 启动一次 Run28 正式训练，默认 `resume=auto`。Codex 未启动且不得代替用户启动本次正式训练。

正式模型锁定前及 Run28 训练过程中，禁止运行 test 或外部候选评价。
