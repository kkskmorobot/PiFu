# Run18 正式训练发布门

`PROJECT_RUN18_GATE=PASS`

`RUN18_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 当前判断

Run18 的静态编译、延迟早停单元测试、CPU/GPU 完整中断恢复、GPU AMP、本地 `tools/yolo26n.pt` 自检、YOLO26m batch=16 容量、真实开发 batch 前向和正式 Trainer 构造均已通过。正式训练结果目录尚未创建。

配置 SHA-256：`e0f6a2154d7bd8cfe85a7eb6f2ebcc4e5458d597832780450bd6cde29108265b`。

ready 指纹：`2ed64c4aa90c6d90d15ace2158de0866e570f71ad8a3cb3bbc3e2b7d0fd2b7a3`。

## 固定边界

- 正式模型：`models/yolo26m-sem.pt`；
- AMP 自检资源：`tools/yolo26n.pt`，SHA-256=`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；
- 早停从可见 epoch 40 开始，patience=30，最早可见 epoch 70 停止；
- 效用门前 fitness=`joint_score-2`，效用门后恢复原始风险约束 fitness；
- batch=16、imgsz=640、epochs=100、AdamW、AMP、seed=42；
- 禁止 test 与外部候选评价。

## 用户授权

2026-08-10，用户明确回复“确认启动 Run18 正式训练”。发布门现已放行；正式训练只能从 PyCharm 启动唯一入口 `tools/run18/train_run18_from_pycharm.py`，不得通过预检脚本或控制台替代正式入口。
