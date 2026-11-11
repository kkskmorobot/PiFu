# Run17 正式训练发布门

`PROJECT_RUN17_GATE=PASS`

`RUN17_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 用户授权

2026-08-10，用户明确回复“确认启动 Run17 正式训练”，并授权使用其电脑中的 PyCharm 启动训练。

根据随后的论文消融讨论，正式 Run17 定义为 YOLO26m 无注意力多任务基线；旧 Run17 边界注意力草案从未训练，已完整归档并作废旧 ready marker。该定义与用户提出的“换模型后重新训练无模块版与加模块版”目标一致，Run16 保留为加模块版。

## 放行对象

- 唯一入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run17_from_pycharm.py`；
- 唯一实验目录：`experiments/17_YOLO26m_Binary8Class_PartialLabel_Baseline`；
- 配置 SHA-256：`f9d684f31194ad98cc2e3560b0299327bc409f4a6543a109d06a8b4f63c6f2c9`；
- ready 指纹：`cfd3404510f226a9059ec3cb9408ed95df5580a18da4ced74a416555aee5d4b8`；
- fresh start：`models/yolo26m-sem.pt`；
- batch=16，imgsz=640，epochs=100，patience=20，AdamW，AMP，seed=42。

## 发布判断

数据、受保护输入、模型结构、真实 batch 前向、完整 batch 反向容量、CPU/GPU 中断恢复和正式 Trainer 构造均通过。正式结果目录尚不存在，未运行 test 或外部评价。

本次不提高 batch、不更换 YOLO26l，因为 Run17 必须与 Run16 保持除注意力外全部因素相同。显存容量优化和更大主干若需要，必须作为后续独立预注册因素。
