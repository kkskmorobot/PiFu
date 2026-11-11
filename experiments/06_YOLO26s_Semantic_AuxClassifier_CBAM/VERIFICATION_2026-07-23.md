# Run6 启动与验证记录（2026-07-23）

## 结论

Run6 已完成代码、结构、损失、验证器和最小 GPU 烟雾验证，并已从 PyCharm 启动正式训练。正式训练只使用 `train` 与 `val`；未读取 `test`。

## 唯一实验变量

相对 Run5，仅在原 P3/8 融合输出之后增加一个标准 CBAM（Channel Attention 后接 Spatial Attention）。数据、预训练权重、优化器、损失、采样、增强、seed 与训练超参数均保持不变。

## 已通过的验证

- 语法编译通过。
- 数据 dry-run 通过：train 8512，val 1504；仅做结构检查。
- 模型结构与损失前后向通过：Run5 参数量 6,550,487；Run6 参数量 6,552,633；CBAM 参数增量 2,146，且仅有一个 CBAM。
- 1% 数据、1 epoch GPU 烟雾训练与完整 val 通过；未运行 test。
- 严格确定性 CUDA backward 通过。CBAM 的 max 描述子采用与全局最大池化等价的 `amax`，避免 CUDA 非确定性 backward。
- 烟雾权重可重新加载，得到语义掩码与 7 类图像分类概率。

## 正式训练启动证据

- 启动入口：`tools/train_skin7_yolo26_cbam.py`
- 配置：`experiments/06_YOLO26s_Semantic_AuxClassifier_CBAM/train_yolo26s_sem_cls_cbam_v6.yaml`
- 结果目录：`experiments/06_YOLO26s_Semantic_AuxClassifier_CBAM/results/semantic_multitask_cbam/skin7_yolo26s_sem_auxcls_cbam_seed42`
- `args.yaml` 已落盘：epochs=100，batch=16，imgsz=640，device=0，seed=42，deterministic=true，split=val。
- PyCharm 控制台已进入 epoch 1/100；首次观察时进度 58/532（约 10%），GPU 显存约 5.64 GB。
- 进程继续在 PyCharm 中运行，不由本次记录等待至训练结束。

## 中断恢复

若训练中断，从 PyCharm 运行 `tools/resume_yolo26_cbam_v6.py`。恢复使用正式目录中的 `weights/last.pt`；请注意 Ultralytics 8.4.101 会恢复优化器、调度器、AMP scaler、EMA 和全局 best fitness，但 EarlyStopping 的历史耐心计数不会完整恢复。

## 选择协议

训练期间保存标准综合指标 `best.pt`、最高图像 Macro-F1 的 `best_macro_f1.pt`，以及满足预注册约束时的 `best_feasible_macro_f1.pt`。正式模型选择前不使用 test。
