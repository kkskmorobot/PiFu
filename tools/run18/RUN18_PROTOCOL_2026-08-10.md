# Run18 预注册协议：YOLO26m 无注意力基线的延迟效用门控早停

`RUN18_PROTOCOL_STATUS=PREREGISTERED`

## 研究目的

Run17 的普通 YOLO26m 多任务基线在可见 epoch 26 早停，而 stopper 的最佳点停留在 epoch 6。该点虽然冷启动风险值较低，但分类效用尚未成熟；后续 epoch 20–25 的二值分割、分类与联合指标明显更好，却没有被原始 fitness 早停规则正确保留。

Run18 不改变模型结构，用一次受控重训修正训练生命周期。它也是下一步 Run19（相同训练生命周期 + Run16 注意力模块）的无注意力对照。

## 唯一预注册因素

相对正式 Run17，唯一变化是早停与 Ultralytics `best.pt` 的 fitness 生命周期：

- epoch 1–39 不累计早停 patience；
- 从可见 epoch 40 开始启用 patience=30，因此最早只能在 epoch 70 早停；
- 未达到 binary IoU≥0.70 且 group Macro-F1≥0.50 时，早停 fitness=`joint_score-2`，从数值区间上保证其低于任何成熟 epoch；
- 达到上述效用门后，早停 fitness 恢复为原始 `joint_score - MEL→NV 超限惩罚`；
- 统一 val 指标、风险约束、`best_binary_iou`、`best_group_macro_f1`、`best_joint`、`best_feasible_joint` 等选择别名完全不变。

模型、分割/分类头、数据、split、partial-label 规则、taxonomy、损失、采样、优化器、学习率、增广、batch=16、imgsz=640、epochs=100、AMP、seed=42 和断点合同均保持 Run17 不变。

## 模型文件边界

- 正式训练模型固定为 `models/yolo26m-sem.pt`；
- `tools/yolo26n.pt` 仅供 Ultralytics CUDA AMP 一致性自检；
- 正式入口在调用 Trainer 前将工作目录固定为 `tools`，并验证该小模型的 SHA-256，防止自动下载到 `tools/run18` 或项目根目录；
- 不把 `yolo26n.pt` 当作正式主干，也不改变 Run17/Run16 的 YOLO26m 对照口径。

## 数据边界

- train/val 仅使用冻结的 development manifest；
- IMA++ source test 634 对完全排除；
- ISIC 2018 Task 1 incremental 与 PAD-UFES-20 仅为外部候选，不参与训练、选择或阈值拟合；
- 正式模型锁定和用户新授权前禁止运行 test 或外部评价。

## 训练与恢复

- PyCharm 入口：`tools/run18/train_run18_from_pycharm.py`；
- fresh start：`models/yolo26m-sem.pt`；
- 默认 `resume=auto`，仅接受大小、SHA-256、指纹和 `_SUCCESS` 均通过的完整 epoch bundle；
- bundle 继续保存 live model、EMA、optimizer、scaler、scheduler、自定义 stopper、selection、Python/NumPy/Torch CPU/CUDA RNG 与数据顺序状态。

## 后续判定

Run18 完成或按新规则正常早停后，只使用统一 val 与 Run17/Run16 比较。若 Run18 的学习曲线正常，再以完全相同的早停策略准备 Run19 注意力版；本实验不运行 test 或外部评价。
