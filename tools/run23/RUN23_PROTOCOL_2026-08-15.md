# Run23 预注册协议：P3/P4 ECA-only seed43 配对重复

`RUN23_PROTOCOL_STATUS=PREREGISTERED`

## 目的

Run23 与 Run22 组成 seed43 下的注意力模块严格配对。Run18/Run21 是 seed42 的 plain/ECA 配对；两组配对共同用于估计 ECA 净效应与随机种子波动，服务于“基于注意力机制改进 YOLO”的毕业论文消融证据。

## 唯一预注册因素

相对 Run22，只在分类头原有 P3、P4 特征上各加入一个 kernel size 5 的 ECA 通道门，然后执行不变的 global-average pooling 与 MLP：

- ECA 一维卷积权重全零初始化，使初始乘法门严格等于 1；
- ECA 仅由分类损失学习；
- 不使用病灶空间 mask，不增加 P5 描述符，不修改语义分割分支；
- 相对 Run21 仅把 seed 从 42 改为 43，用于与 Run22 配对。

以下全部保持 Run22 不变：

- YOLO26m-sem 预训练权重；
- development train/val manifest、partial-label loader 与 split；
- 损失、采样、增广、batch=16、imgsz=640、AdamW 与学习率策略；
- seed=43、deterministic=true；
- 100 epochs、epoch40 后启用 patience=30，最早可见 epoch70 早停；
- 统一 val、joint-score-2、mel→nv 风险约束与 checkpoint v2 合同。

## 数据和评价边界

只允许 development train/val。IMA++ test、ISIC 2018 Task 1 外部候选和 PAD-UFES-20 均不得用于训练、选择、阈值拟合或本阶段评价。

## 训练入口

唯一入口：`tools/run23/train_run23_from_pycharm.py`。默认 `resume=auto`。正式实验目录必须在 PyCharm 启动前不存在。

## 后续决策

Run23 完整结束后，报告 Run21-Run18 与 Run23-Run22 两个同 seed 配对差值。若符号或幅度不稳定，优先补第三组 plain/ECA seed，而不是立即增加新模块、修改风险阈值或读取 test。
