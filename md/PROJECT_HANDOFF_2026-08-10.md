# PiFuV2 项目交接（2026-08-10）

## 当前状态

- Run1–Run16 的正式结果必须保留；Run16 已正常早停完成；
- Run16 结果结论见 `PROJECT_RUN16_RESULT_2026-08-10.md`；
- 正式 Run17 已重新预注册为 YOLO26m 无注意力 partial-label 多任务基线，用于与 Run16 做同主干注意力消融；
- 用户已确认并授权通过 PyCharm 启动 Run17；
- 正式模型选择前禁止运行 test；ISIC 2018 Task 1 incremental 与 PAD-UFES-20 不得用于训练、选择或阈值拟合。

## Run16

- 正常早停：CSV 可见 epoch 61，最终完整 bundle 为 epoch 60；
- 最佳 binary IoU/Dice 0.77389/0.87254；group Macro-F1 0.60160；joint 0.68049；
- 没有 checkpoint 满足 MEL→NV≤0.1077844311；
- Run16 不废弃，作为 YOLO26m + BLG-MDA 加模块版本保留。

## Run17

- 正式入口：`train_run17_from_pycharm.py`；
- 正式目录：`experiments/17_YOLO26m_Binary8Class_PartialLabel_Baseline`；
- config SHA-256：`f9d684f31194ad98cc2e3560b0299327bc409f4a6543a109d06a8b4f63c6f2c9`；
- ready fingerprint：`cfd3404510f226a9059ec3cb9408ed95df5580a18da4ced74a416555aee5d4b8`；
- 单一因素：相对 Run16 删除 P3/P4/P5 ECA、detached 病灶池化和 P5 描述符，恢复 P3/P4 global-average plain classifier；
- 其余训练、数据、val、checkpoint 与选择协议全部保持不变；
- batch=16 的完整 AMP forward/backward/AdamW 峰值 reserved 约 6.996 GB（设备的 40.93%），容量充足；为了消融可比性本轮不扩大 batch、不换 YOLO26l；
- CPU 与 GPU AMP workers=2 的中断恢复逐状态差异均为 0；正式构造器 v2 PASS；
- release gate 已记录用户确认并设为 PASS。

## 旧边界草案

较早的 `experiments/17_YOLO26m_Binary8Class_BinaryLesionBoundaryGuidedAttention` 仅为预检草案，从未训练。它已标记 superseded，11 个相关文件复制归档至 `preflight/superseded_run17_boundary_draft_20260810_v1`。如果未来继续边界注意力，必须使用新的实验编号。

## 训练完成后的固定流程

1. 先核验 completion marker、CSV 连续性、完整 bundle 与恢复指纹；
2. 仅使用统一 val 协议比较 Run17 与 Run16；
3. 判断注意力模块的净增益并决定是否做后续边界变体或多随机种子确认；
4. 在正式模型锁定和用户新授权前，不运行 test 或外部评价。
