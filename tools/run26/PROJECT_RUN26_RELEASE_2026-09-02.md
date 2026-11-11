# Run26 正式训练发布门

`PROJECT_RUN26_GATE=PASS`

`RUN26_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

Run26 已预注册为 Run25 的 P3/P4 ECA + 恒等初始化空间注意力单因素实验，全部技术预检已通过。

- 配置 SHA-256：`60f24b15ec51995f9728e86777bda3dcf6274bb7feac5cfd92652ed7e4500546`
- 就绪指纹：`a440204a90d71e79e1d1c9cfa25fadcdc7a0c76854db469a01387bae6451abd7`
- 静态与回归测试：`8 passed`
- CPU 与 GPU/AMP/双 worker 断点恢复：连续训练和恢复训练状态逐项零差异
- RTX 5080 容量预检：batch=16、imgsz=640、AMP、AdamW 单步通过，峰值 reserved 显存占比约 41.47%
- prepare-only 与正式 Trainer 构造：通过，未调用 `trainer.train()`
- 正式 Run26 实验目录：尚未生成

用户已于 2026-09-02 明确回复“确认启动 Run26 正式训练”，确认已记录，`PROJECT_RUN26_GATE=PASS`。允许从锁定的 PyCharm 入口脚本启动一次 Run26 正式训练。

正式训练前及训练过程中禁止运行 test 或外部候选评价。
