# Run34 发布门（2026-09-10）

Run34 是 development-only 固定代价决策评估，不是训练。

规范入口为 `evaluate_run34_from_pycharm.py`；原 `run_run34_from_pycharm.py` 保留为兼容转发入口。正式评估使用 bundle `runtime_state.pt` 内的无损 fp32 EMA，并复现训练期未融合/AMP validation 数值路径。

`PROJECT_RUN34_GATE=PASS`
`RUN34_READY_TO_CLICK_GATE=PASS`
`USER_CONFIRMATION_RECORDED=TRUE`
`TRAINING_PERFORMED=FALSE`
`TEST_METRICS_USED=FALSE`
`EXTERNAL_METRICS_USED=FALSE`
`PAD_UFES_20_USED=FALSE`

用户已要求在 Run33 完成后依据根目录规划选择并准备下一步。根据 Run33 失败的预注册 CBAM 复现门，本次不再准备新的模型训练，而是准备 Run34 冻结 checkpoint 决策研究。
