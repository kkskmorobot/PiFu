# Run22 预注册协议：Run18 plain 的 seed43 配对重复

`RUN22_PROTOCOL_STATUS=PREREGISTERED`

## 目的

Run21 ECA-only 相对 Run18 的最佳 Joint 只高 0.00039，必须先测量随机种子波动，再判断注意力净增益。Run22 是未来 Run23 ECA seed43 的严格无模块配对基线。

## 唯一预注册因素

相对 Run18，只把 Ultralytics 随机种子从 `42` 改为 `43`。实验编号、输出路径和契约标识随新实验更新，不属于训练行为因素。

以下全部保持 Run18 不变：

- YOLO26m-sem 预训练权重；
- P3/P4 plain global-average-pooling 分类头，无注意力、无P5；
- 冻结 development train/val manifest 和 partial-label loader；
- 损失、采样、增广、batch=16、imgsz=640、AdamW 与学习率策略；
- 100 epochs、epoch40后启用 patience=30；
- 统一 val、joint-score-2、风险约束和 checkpoint v2 合同。

## 数据和评价边界

只允许 development train/val。IMA++ test、ISIC 2018 Task 1 外部候选和 PAD-UFES-20 均不得用于训练、选择、阈值拟合或本阶段评价。

## 训练入口

唯一入口：`tools/run22/train_run22_from_pycharm.py`。默认 `resume=auto`；最早可见 epoch 70 早停，最多100 epochs。

## 后续配对

Run22 完成并通过完整性复核后，才准备 Run23 ECA-only seed43。Run22/Run23 的比较因素将仅为 ECA 模块。
