# PiFuV2 Run14 预注册协议

日期：2026-08-09  
主基线：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`  
诊断依据：Run11 与 Run13  
实验目录：`experiments/14_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVLogitMargin`

## 1. 触发依据

Run13 完成可见 epoch 1~72 后正常早停，6 个保留 bundle 完整，auto resume 指向内部
epoch 71，无异常或中断。最佳 image Macro-F1、group Macro-F1、joint 为
0.53763、0.59183、0.67586，均低于 Run12；最低 MEL->NV=0.15306，虽优于 Run12
的 0.25510，但差于无注意力 Run11 的 0.11224，且仍高于冻结上限 0.1077844311。

因此 Run13 的“BLG-MDA attention + 全 MEL target CE 权重”存在不利交互：没有保住
Run12 的整体优势，也没有复现 Run11 的风险改善。继续已正常早停的 Run13不合理。

## 2. 唯一变化因素

Run14 回到 Run12 的 unweighted CE 和 BLG-MDA 架构，只加入一个训练期、目标条件式的
MEL-vs-NV competitor logit margin：

- 仅当真实分类 target 为 MEL 时，在计算 CE 前给该样本的 NV logit 加上 margin；
- margin=`ln(train_n_nv/train_n_mel)=ln(5539/3047)=0.5976564790473948`；
- 这迫使模型在 MEL 样本上学习更大的 `mel_logit - nv_logit` 间隔；
- 其他 target 的 logits 完全不改；
- validation、inference、argmax、temperature=1 的 logits 完全不改；
- margin 只由 train-only 计数推导，没有使用 val/test/external 调参。

该因素简称 TMNCM（Target-conditioned MEL-NV Competitor Margin）。相对 Run12，唯一
行为变化是分类训练 CE 内的上述固定 margin。

## 3. 严格保持不变

- Run12 development manifest 逐字节复制，train/val=23,462/1,744；
- source test 634 行排除，YAML `test: null`，external 不进入 manifest；
- fresh `models/yolo26s-sem.pt`，不从 Run12/13 best 或 last warm-start；
- YOLO26s、binary semantic main/aux、P3/P4/P5 BLG-MDA、detached lesion map、
  identity ECA、mask mix=0.5 全部不变；
- image classification loss 总系数=0.5，无 class weight、无 sampler、无 focal/asymmetric
  loss、无 threshold/temperature fitting；
- partial-label loader、ignore mask、统一 validator、选择与风险阈值不变；
- batch=16、imgsz=640、epochs=100、patience=20、AdamW、LR、增强、AMP、seed=42；
- test、ISIC2018 external、PAD-UFES-20 不用于训练、选择或超参数决策。

## 4. 冻结评价与选择

- `joint_score=sqrt(binary_iou * group_macro_f1)`；
- feasible iff group MEL->NV rate <= 0.1077844311 且 MEL support 存在；
- primary alias=`best_feasible_joint`；
- 无 feasible checkpoint 则 Run14 判约束失败，无约束 best 仅作诊断；
- 必须报告 segmentation、image/group 分类、双向 MEL/NV 风险、恶性转良性、NLL、
  Brier、ECE、参数量、GFLOPs 和速度；
- 正式模型锁前禁止 test，external 仍只作候选。

## 5. 完整恢复与启动边界

每个完成 epoch 原子发布 `_SUCCESS` bundle，保存 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、stopper、selection、全部 RNG 和 sampler/data progress；
auto/explicit resume 必须校验大小、SHA-256、fingerprint 与完整状态。

只有 Run14 静态/回归、CPU/GPU 完整恢复、正式 Trainer 构造和真实 development batch
预检全部 PASS、最新 release 明确 `PROJECT_RUN14_GATE=PASS`，并由用户再次确认后，
才可从 PyCharm 运行 `train_run14_from_pycharm.py`。准备阶段禁止 test/external。
