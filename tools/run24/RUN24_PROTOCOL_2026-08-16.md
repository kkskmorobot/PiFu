# Run24 预注册协议：plain seed44 第三配对基线

`RUN24_PROTOCOL_STATUS=PREREGISTERED`

## 目的

Run18/Run21 与 Run22/Run23 的 ECA 配对结果在 Joint、Binary IoU 和 Image Macro-F1 上方向不一致。Run24 建立第三个 plain 随机种子基线，为未来 Run25 ECA seed44 提供严格同 seed 对照。

## 唯一预注册因素

相对 Run22，只把 Ultralytics 随机种子从 `43` 改为 `44`。实验编号、输出路径和合同标识随新实验更新，不属于训练行为因素。

以下全部保持 Run22 不变：

- YOLO26m-sem 预训练权重；
- P3/P4 plain global-average-pooling 分类头，无注意力、无 P5；
- development train/val manifest 与 partial-label loader；
- 损失、采样、增广、batch=16、imgsz=640、AdamW 与学习率策略；
- 100 epochs、epoch40 后启用 patience=30；
- 统一 val、joint-score-2、mel→nv 风险约束与 checkpoint v2 合同。

## 数据和评价边界

只允许 development train/val。IMA++ test、ISIC 2018 Task 1 外部候选和 PAD-UFES-20 均不得用于训练、选择、阈值拟合或本阶段评价。

## 训练入口

唯一入口：`tools/run24/train_run24_from_pycharm.py`。默认 `resume=auto`；最早可见 epoch70 早停，最多 100 epochs。

## 后续配对

Run24 完成并通过完整性复核后，才准备 Run25 ECA-only seed44。完成第三组后，以三个同 seed 配对差值汇总 ECA 的均值、离散程度和方向一致性，再决定保留、废弃或改造注意力模块。
