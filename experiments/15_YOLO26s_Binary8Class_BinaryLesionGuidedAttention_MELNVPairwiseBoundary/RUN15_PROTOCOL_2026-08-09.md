# PiFuV2 Run15 预注册协议

日期：2026-08-09  
主基线：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`  
诊断依据：Run11、Run13、Run14  
实验目录：`experiments/15_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVPairwiseBoundary`

## 1. 触发依据

Run14 在可见 epoch 62 正常早停，训练与 6 个完整 bundle 均通过验收。其最佳 group
Macro-F1/joint=0.58802/0.67369，低于 Run12 的 0.59969/0.68258；成熟最佳 joint
状态 MEL->NV=0.35714。唯一阈值内点来自可见 epoch 1，但 group Macro-F1=0.16923、
binary IoU=0.53363，是冷启动退化状态，不具备临床任务效用。

Run14 把 NV margin 放进八分类 CE；当其他六类 logit 占主导时，MEL-NV 的直接梯度
会被 softmax 概率稀释。继续已正常早停的 Run14不能修复该目标函数问题。

## 2. 唯一训练因素：DPMNB

Run15 回到 Run12 的 BLG-MDA attention、unweighted eight-class CE，并只增加 Direct
Pairwise MEL-NV Boundary（DPMNB）训练项。对每个真实 MEL 分类样本：

`softplus(NV_logit - MEL_logit + ln(5539/3047))`

- margin=`0.5976564790473948`，仅由 train MEL/NV 计数推导；
- unit coefficient=1.0，不进行 val/test/external 调参；
- 对全部有分类标签样本数归一化，使每个 MEL 样本贡献稳定且无 MEL batch 为精确 0；
- base CE 对全部八类保持 unweighted；其他 target 不产生 pairwise gradient；
- validation、inference、argmax、temperature=1 logits 完全不改；
- 相对 Run12，初始化、参数量和推理结构逐元素一致。

该直接二元边界梯度只作用于 MEL/NV 两个 logits，不会被其他六类 logit 稀释。

## 3. 严格保持不变

- Run12 development manifest 逐字节复制，train/val=23,462/1,744；
- source test 634 行排除，YAML `test: null`，external 不进入 manifest；
- fresh `models/yolo26s-sem.pt`，不 warm-start Run12/14；
- YOLO26s、binary semantic main/aux、P3/P4/P5 BLG-MDA、detached lesion map、
  identity ECA、mask mix=0.5 全部不变；
- image classification总系数=0.5，无 class weight、无 sampler、无 focal/asymmetric；
- partial-label loader、ignore mask、batch=16、imgsz=640、epochs=100、patience=20、
  AdamW、LR、增强、AMP、seed=42 全部不变；
- test、ISIC2018 external、PAD-UFES-20 不用于训练、选择或超参数决策。

## 4. 非退化统一选择协议

- `joint_score=sqrt(binary_iou * group_macro_f1)`；
- `risk_feasible` iff MEL support 存在且 group MEL->NV<=0.1077844311；
- `utility_eligible` iff binary IoU>=0.70 且 group Macro-F1>=0.50；
- `feasible_joint` iff risk_feasible 且 utility_eligible；
- primary alias=`best_feasible_joint`，按 joint 最大选择；
- early-stopping fitness 仍为原 joint 减 MEL->NV 超限量，协议修正不改变训练 fitness；
- 无合格 checkpoint 则判约束失败，无约束 best 仅作诊断；
- 必须报告 segmentation、image/group 分类、双向 MEL/NV 风险、恶性转良性、NLL、
  Brier、ECE、参数量、GFLOPs 和速度。

## 5. 完整恢复与启动边界

每个完成 epoch 原子发布 `_SUCCESS` bundle，保存 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、stopper、selection、全部 RNG 和 sampler/data progress；
auto/explicit resume 必须验证大小、SHA-256、fingerprint 和完整状态。

只有 Run15 静态/回归、CPU/GPU 完整恢复、正式 Trainer 构造和真实 development batch
预检全部 PASS，最新 release 明确 `PROJECT_RUN15_GATE=PASS`，并由用户再次确认后，
才可从 PyCharm 运行 `train_run15_from_pycharm.py`。准备阶段禁止 test/external。
