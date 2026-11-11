# Runs5 最终验证与 experiments/06 决策

日期：2026-07-23  
数据范围：仅 `val`，未读取 `test`。  
统一验证：`tools/evaluate_skin7_unified.py`，lesion 分层 bootstrap 2000 次，seed=42。

## 1. 训练完成状态

- 正式训练日志完整到第 84 轮，随后早停并正常完成最终验证。
- `best.pt` 和 `last.pt` 已被 Ultralytics 正常 strip，大小约 13.3 MB。
- 训练曾在第 30 轮后中断并从 `last.pt` 恢复。
- checkpoint 恢复了 optimizer、scheduler、scaler、EMA 和 global best fitness；Ultralytics 8.4.101 没有序列化 EarlyStopping 的 `best_epoch`/patience counter，因此恢复后早停计数重新开始。这解释了训练继续到第 84 轮，但不改变已记录的逐轮 val 指标。

## 2. checkpoint 选择审计

预设复合分数为：

`0.50 * Macro-F1 + 0.25 * foreground-mIoU + 0.25 * minority-recall - mel→nv excess penalty`

它把第 18 轮选为 `best.pt`，score=0.61133。该点少数类 Recall 较高，但 foreground-mIoU 只有 0.4696，违反“分割不明显低于 runs4”的原始约束。第 64 轮日志具有更高的图像 Macro-F1（0.6980）和更好的 foreground-mIoU（0.4992），但复合分数仍略低于第 18 轮，而且没有对应逐轮 checkpoint。

因此，runs5 的 `best.pt` 只能称为“复合分数最佳”，不能称为满足论文全部约束的最终模型。后续实验必须同时保留逐项指标与 Pareto 判断，不能只看 `best.pt`。

## 3. runs4 与 runs5 统一验证对比

| 模型 | foreground mIoU | Image Acc | Image Macro-F1 | 少数类 Recall | mel→nv | nv→mel | Lesion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| runs4 best | 0.538748 | 0.809176 | 0.704120 | 0.533498 | 18 | 84 | 0.724283 |
| runs5 best（第18轮） | 0.470054 | 0.751330 | 0.672016 | 0.631691 | 10 | 63 | 0.670156 |
| runs5 epoch60 | 0.505156 | 0.777261 | 0.683651 | 0.546348 | 13 | 161 | 0.724404 |

### runs5 best 的变化

- 有利：`mel→nv` 18→10，`nv→mel` 84→63；akiec Recall 0.449→0.612，df Recall 0.333→0.556。
- 不利：foreground-mIoU 下降 0.0687；Image Macro-F1 下降 0.0321；nv Recall 0.844→0.751；bcc Recall 0.818→0.727。
- 结论：早期分类头提高了少数类敏感度和部分风险错误，但以整体准确率、Macro-F1 和分割质量为代价。

### runs5 epoch60 的变化

- 分割和总体分类比 runs5 best 更平衡，Lesion Macro-F1 与 runs4 几乎相同。
- Image Macro-F1 仍低于 runs4（0.6837 vs 0.7041），foreground-mIoU 仍低 0.0336。
- `mel→nv` 改善为 13，但 `nv→mel` 恶化到 161；akiec Recall 降到 0.265。
- lesion Macro-F1 的 bootstrap 区间与 runs4 明显重叠，不能宣称稳定提升。

## 4. 过拟合诊断

训练后期 `train/image_cls_loss` 已降至约 0.007，而 `val/image_cls_loss` 升至约 1.84；语义 val CE 同时升至约 1.25。辅助分类头出现明显过拟合，并且共享特征上的分类梯度可能损害语义分割。这个现象应记录为 runs5 的正式结论，而不是通过事后改 loss 隐藏。

## 5. experiments/06 决策

experiments/06 仍按原计划做 **runs5 + 标准 CBAM**，而不是现在改数据、采样、分类 loss 权重或 focal loss。原因：

1. runs5 已完成“显式分类头本身是否有效”的必要控制，结果是部分风险指标改善但总体未胜过 runs4。
2. 论文仍需要一个已知通用注意力模块作为 runs7 提出方法的公平对照。
3. 若此时先改 loss/sampler，后续无法区分收益来自注意力还是优化策略；这些保留给 runs8。

### runs6 唯一结构变化

- 在 YOLO26 semantic neck 的 P3/8 融合输出（原模型 layer 16）后插入 **一个标准 CBAM**。
- 顺序：Channel Attention → Spatial Attention。
- semantic head 和 P3/P4 图像分类头继续使用 CBAM 后的 P3 与原 P4 特征。
- 不在 backbone/P4/P5 多处堆叠 CBAM，避免参数增加和归因不清。

### runs6 保持不变

- V3 数据、split、seed=42、batch=16、imgsz=640。
- AdamW、lr0=0.001、cosine schedule、增强策略、100 epochs、patience=20。
- semantic loss 与 `image_cls_weight=0.5`。
- 不使用 test，不改变 sampler，不加 focal/balanced-softmax，不加入拟议的 lesion-guided attention。

### runs6 预注册判断

- 首要比较：Image Macro-F1 相对 runs5 同协议 checkpoint 是否提高，并同时报告 runs4。
- 约束：foreground-mIoU、`mel→nv`、少数类 Recall 分开报告；不再让复合分数掩盖约束失败。
- 每轮保存或至少保存足够密集的 checkpoint，正式结论基于预注册的 Pareto 规则：优先 Macro-F1，要求风险不劣于 runs4，分割下降必须明确量化。
- 如果没有 checkpoint 同时满足约束，则把 runs6 判为“标准 CBAM 无稳定收益”，不要事后更改阈值。

## 6. 后续路线

- runs6：runs5 + 单点标准 CBAM。
- runs7：回到 runs5 主干，加入 lesion-mask-guided multi-scale dual attention，作为论文提出方法。
- runs8：仅在结构确定后做 lesion-balanced sampler / loss 优化消融。

