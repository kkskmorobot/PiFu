# Run25 预注册协议：P3/P4 ECA-only seed44 配对重复

`RUN25_PROTOCOL_LOCKED=TRUE`

`TEST_POLICY=FORBIDDEN_UNTIL_FINAL_MODEL_LOCK_AND_FRESH_USER_AUTHORIZATION`

`EXTERNAL_POLICY=FORBIDDEN_FOR_TRAINING_SELECTION_AND_THRESHOLD_TUNING`

## 研究问题

Run25 与 Run24 组成 seed44 下的第三组注意力模块严格配对。Run18/Run21 为 seed42 plain/ECA，Run22/Run23 为 seed43 plain/ECA。三组配对用于估计 ECA 的净效应与随机种子波动，服务于“基于注意力机制改进 YOLO 的皮肤病识别算法研究”的毕业论文消融证据。

## 唯一预注册因素

相对 Run24，只在分类头原有 P3、P4 特征上各加入一个 kernel size 5 的 ECA 通道门，再执行不变的 global-average pooling 与 MLP：

- ECA 一维卷积权重全零初始化，使初始乘法门严格等于 1；
- ECA 只由分类损失学习；
- 不增加空间注意力，不增加 P5 描述符，不修改二值语义分割分支；
- 相对 Run23 只把 seed 从 43 改为 44，用于与 Run24 配对。

## 冻结训练合同

- 初始化：`models/yolo26m-sem.pt`；AMP 本地校验资产：`tools/yolo26n.pt`；禁止运行时下载；
- 数据：既有 partial-label development train/val；test 为 null，PAD-UFES-20 和 ISIC 2018 外部候选不得进入训练或选择；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、seed=44、deterministic=true、AMP=true；
- loss、数据采样、增强及分类头 MLP/dropout 与 Run24 完全一致；
- 早停从 epoch 40 后才计入，patience=30，最早停止点为 epoch 70；
- 主选择别名为 `best_feasible_joint`；mel→nv≤0.1077844311、Binary IoU≥0.70、Group Macro-F1≥0.50；
- 若无风险可行点，只报告 constraint failure 和 best joint 诊断点，不把它冒充正式可行候选。

## 启动与恢复

唯一正式入口为 `tools/run25/train_run25_from_pycharm.py`，默认 `resume=auto`。正式实验目录必须在首次 PyCharm 启动前不存在。只有 release 文档记录用户确认并改为 PASS 后才允许点击正式训练。

## 完成后的决策

Run25 完成后报告 Run21−Run18、Run23−Run22、Run25−Run24 三个同 seed 配对差值，并汇总均值、离散性和方向一致性。正式模型锁定前仍不得读取 test 或外部候选评价。
