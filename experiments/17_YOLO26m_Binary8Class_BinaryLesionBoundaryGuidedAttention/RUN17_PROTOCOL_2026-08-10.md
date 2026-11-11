# PiFuV2 Run17 预注册协议

日期：2026-08-10  
主基线：`experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention`  
实验目录：`experiments/17_YOLO26m_Binary8Class_BinaryLesionBoundaryGuidedAttention`

## 1. Run16 触发依据

Run16 在可见 epoch 61 正常早停，61 行连续且非有限值为 0，5 个保留 bundle 均通过
完整性验证。最佳 group Macro-F1=0.60160，略高于 Run12 的 0.59969；最佳 joint=
0.68049，略低于 Run12 的 0.68258。更重要的是，Run16 的成熟最低 MEL->NV 为
0.34694（group Macro-F1=0.59534、joint=0.67794），优于 Run12 的 0.40816，说明
YOLO26m 改善了风险—效用前沿，但仍没有达到 0.1077844311 的约束。

因此 Run16 作为容量主效应实验正常结束，不续训。下一步保留其 YOLO26m 容量，回到
本项目“基于注意力机制改进 YOLO”的主线，增强对皮损边界不规则性和色素扩散区域的
显式聚合，而不是再次改变类别损失。

## 2. 唯一训练因素

相对 Run16，唯一训练行为变化是 BLBG-MDA 边界描述子：

- 从不变的 binary semantic main logit 得到 detached sigmoid lesion probability；
- 用水平/垂直一阶差分绝对值之和生成 detached soft lesion-boundary map；
- boundary map 逐图最大值归一化，并在 P3/P4/P5 上做 normalized boundary-weighted average；
- 冻结的 Run16 global+lesion 三尺度描述子保持原样，边界三尺度描述子追加在其后；
- 分类 MLP 第一层新增的所有 boundary columns 精确零初始化，旧列、bias 和输出层逐元素
  继承 Run16 构造，因此初始 semantic 输出与 8-class logits 均和 Run16 一致；
- boundary map 对 semantic branch detach，分类损失不增加回到分割头的新梯度路径。

该单一架构因素简称 BLBG-MDA
（Binary-Lesion-and-Boundary-Guided Multi-scale Dual Attention）。

## 3. 严格保持不变

- 官方 `models/yolo26m-sem.pt` fresh start 及其 SHA-256；
- development manifest/split/mask/partial-label scope 与 Run16 逐字节一致；
- train/val=23,462/1,744，test 634 行排除，development YAML 仍为 `test: null`；
- semantic main/aux head、binary loss 和分割监督 scope；
- P3/P4/P5 identity-initialized ECA、50% global + 50% lesion-guided base pooling；
- unweighted 8-class CE、image loss weight=0.5、label smoothing=0；
- batch=16、imgsz=640、epochs=100、patience=20、AdamW、LR、增强、AMP、seed=42；
- 标准 seeded shuffle，每轮所有 23,462 个 train 样本一次；
- Run13/14/15 的 MEL weight、logit margin、pairwise loss 均不继承；
- checkpoint bundle v2 的完整状态、原子 `_SUCCESS` 与 auto/explicit resume 合同；
- test、ISIC2018 Task 1 和 PAD-UFES-20 不进入训练、选择或阈值调优。

## 4. val-only 判定

- joint=`sqrt(binary_iou * group_macro_f1)`；
- risk limit：group MEL->NV <= 0.1077844311；
- non-degenerate eligibility：binary IoU>=0.70 且 group Macro-F1>=0.50；
- primary alias=`best_feasible_joint`；无 feasible 时只报告诊断最佳点，不建立模型锁；
- early-stopping fitness 继续保持原 `joint - MEL->NV excess`，边界资格规则不改 fitness；
- 必须报告 binary IoU/Dice、image/group Macro-F1、MEL/NV 双向风险、NLL/Brier/ECE、
  参数量、GFLOPs 和速度；正式模型锁前禁止 test/external。

## 5. 启动边界

正式训练只能在实现、真实 development batch、CPU/GPU 完整恢复和正式 Trainer 只构造
全部 PASS，最新 release 明确写出 `PROJECT_RUN17_GATE=PASS` 且用户再次确认后，由用户
从 PyCharm 无参数运行 `train_run17_from_pycharm.py`。准备阶段不得创建正式结果目录，
不得调用 `trainer.train()`，不得运行 test 或 external evaluator。
