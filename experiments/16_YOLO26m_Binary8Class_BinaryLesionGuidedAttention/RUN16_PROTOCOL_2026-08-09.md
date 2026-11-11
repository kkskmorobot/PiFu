# PiFuV2 Run16 预注册协议

日期：2026-08-09  
主基线：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`  
诊断对照：Run13、Run14、Run15  
实验目录：`experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention`

## 1. 触发依据

Run12 在现有数据合同上取得迄今较好的整体 val-only 结果，但没有满足 MEL->NV 风险
上限。Run13 的 MEL weighted CE、Run14 的八分类 logit margin 和 Run15 的 direct
pairwise boundary 均未同时改善整体分类、联合分数与成熟阶段 MEL->NV 风险；Run15
在可见 epoch 58 正常早停，feasible checkpoint 数量仍为 0。因此不继续已经早停的
Run15，也不叠加其损失因素。

本项目研究主题是基于注意力机制改进 YOLO。下一步检验更大预训练语义骨干能否为
已冻结的 BLG-MDA 注意力分类头提供更充分的颜色、纹理和边界特征。

## 2. 唯一训练因素

相对 Run12，唯一训练行为变化为：

- `models/yolo26s-sem.pt` 替换为官方 `models/yolo26m-sem.pt`；
- 冻结权重 SHA-256：
  `3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`。

Run12 的二值语义头、BLG-MDA P3/P4/P5 病灶引导注意力、分类 MLP、unweighted CE
和 partial-label loss 全部保持不变。模型规模增加是本实验唯一预注册训练因素。

## 3. 严格保持不变

- development manifest 的 train/val 行逐字节继承，split、mask 值域和监督可用性不变；
- train/val=23,462/1,744；seg=10,483/292；cls=13,715/1,488；
- batch=16、imgsz=640、epochs=100、patience=20、AdamW、LR、增强、AMP、seed=42；
- 每轮标准 seeded shuffle、所有 23,462 个 train 样本恰好一次；
- image classification loss weight=0.5、unweighted CE、label smoothing=0；
- BLG-MDA 的 ECA、detached binary lesion map、50/50 pooling 和 P5 零贡献初始化；
- 不继承 Run13 MEL weighting、Run14 logit margin 或 Run15 pairwise loss；
- checkpoint bundle v2 的完整状态和 completed-epoch 原子发布合同；
- test、ISIC2018 Task 1 external 和 PAD-UFES-20 external 不进入训练、选择或调参。

## 4. 选择协议

统一 val 指标保持不变：`joint=sqrt(binary_iou * group_macro_f1)`，分类使用 argmax，
分割阈值 0.5。风险可行性上限仍为 group MEL->NV <= 0.1077844311。

保留 Run15 已登记的模型选择纠错：`best_feasible_joint` 还必须满足
binary IoU >= 0.70 且 group Macro-F1 >= 0.50，以排除尚未学会任务的冷启动假可行点。
该规则只影响 feasibility/alias；early-stopping fitness 仍是原先的
`joint - MEL->NV excess`，不改变优化、学习率或训练轨迹。

## 5. 显存与恢复边界

RTX 5080 16GB 上，随机初始化 YOLO26m + BLG-MDA 的 640、batch=16、AMP 合成
forward/backward/AdamW 单步已通过，峰值 allocated/reserved 为
6,463,043,584 / 6,994,001,920 bytes。正式发布前仍必须使用实际预训练权重完成真实
development batch forward/loss、正式 Trainer 只构造以及 CPU/GPU 中断恢复等价性检查。

正式训练只能在最新 Run16 release 明确写出 `PROJECT_RUN16_GATE=PASS`、
`USER_CONFIRMATION_RECORDED=TRUE` 且用户再次确认后，由用户从 PyCharm 无参数运行
`train_run16_from_pycharm.py`。正式模型选择前禁止运行 test 或 external evaluator。
