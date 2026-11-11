# Run19 预注册协议：匹配训练生命周期的 YOLO26m BLG-MDA 注意力实验

`RUN19_PROTOCOL_STATUS=PREREGISTERED`

## 决策依据

Run18 完整训练 100 epochs，证明延迟且效用门控的早停修复了 Run17 的冷启动过早停止。Run18 的最佳 joint=0.68246、group Macro-F1=0.61023；旧 Run16 注意力版在 61 epochs 早停，最佳 joint=0.68049、group Macro-F1=0.60160。差异很小且训练时长不匹配，不能据此判断注意力无效。

因此 Run19 按原计划补齐严格对照：使用 Run18 的全部训练生命周期与数据合同，只恢复 Run16 已冻结的二值病灶引导多尺度注意力（BLG-MDA）。

## 唯一预注册因素

相对 Run18，唯一变化是图像分类头：

- P3/P4 扩展为 P3/P4/P5 描述符；
- 每个尺度加入 identity-initialized ECA 通道门控；
- 由 detached 二值分割主 logit 形成病灶空间权重；
- 全局平均池化与病灶引导池化各占 0.5；
- P5 初始贡献为 0，分割分支不修改。

模型主干、预训练权重、数据、split、partial-label、taxonomy、损失、采样、优化器、学习率、增广、batch=16、imgsz=640、epochs=100、seed=42、AMP、本地 AMP 小模型、epoch 40 后启用的 patience=30、`joint_score-2` 冷启动门控、统一 val 指标、风险约束与 checkpoint 合同全部保持 Run18 不变。

## 模型与数据边界

- 正式模型：`models/yolo26m-sem.pt`；
- AMP 自检：`tools/yolo26n.pt`，只作 Ultralytics 本地一致性检查；
- train/val 只使用冻结 development manifest；
- IMA++ test 634 对、ISIC 2018 Task 1 incremental、PAD-UFES-20 均不参与训练、选择或阈值拟合；
- 正式模型锁定并获得新的用户授权前禁止 test 或外部评价。

## 训练与恢复

- PyCharm 唯一入口：`tools/run19/train_run19_from_pycharm.py`；
- fresh start：`models/yolo26m-sem.pt`；
- 默认 `resume=auto`，只接受完整且指纹一致的 epoch bundle；
- 最早可见 epoch 70 早停；最多训练 100 epochs。

## 结果解释

Run19 与 Run18 构成毕业论文的主注意力消融对照。完成后只用统一 val 比较，不运行 test；若风险约束仍无可行 checkpoint，将其作为独立于注意力消融的分类风险问题报告，不在 Run19 同时改变损失或采样。
