# Run27 正式训练发布门

`PROJECT_RUN27_GATE=PASS`

`RUN27_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run27 已预注册为 Run21 seed42 P3/P4 ECA-only 的恒等初始化 spatial7 单因素配对复现。全部技术预检已通过。

- 配置 SHA-256：`4732315643841f00107c949bf5585df1c9e07e8255f2813d92da825d892a7619`
- 最终就绪指纹：`85880df0bfb882373d0ffe89fcd2f65b5c3c9bf50ecd9b2f5bf1a4c0a57a10fa`
- Run21 配对基线指纹：`b7b853c4284037a6d51d0e66e10cd0302d167bf4ab30c7b32f512f6f137fb1a9`
- 静态与回归测试：`8 passed`
- CPU 与 GPU/AMP/workers=2 断点恢复：连续与恢复状态逐项零差异
- RTX 5080 容量预检：batch=16、imgsz=640、AMP、AdamW 单步通过，峰值 reserved 显存占比约 41.47%
- development `--prepare-only` 与正式 Trainer 构造：通过，均未调用正式 `trainer.train()`
- 正式 Run27 实验目录：不存在

用户于 2026-09-05 明确要求把 Run27 准备到由其本人在 PyCharm 点击训练之前，并明确说明由用户本人点击训练；该确认已记录。允许用户从锁定入口 `tools/run27/train_run27_from_pycharm.py` 启动一次 Run27 正式训练，默认 `resume=auto`。Codex 未启动且不得代替用户启动本次正式训练。

正式模型锁定前及 Run27 训练过程中，禁止运行 test 或外部候选评价。
