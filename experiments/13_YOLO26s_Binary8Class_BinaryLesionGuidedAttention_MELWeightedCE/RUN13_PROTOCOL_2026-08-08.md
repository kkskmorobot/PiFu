# PiFuV2 Run13 预注册协议

日期：2026-08-08  
主基线：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`  
因素参照：Run10（无注意力/无 MEL 权重）、Run11（无注意力/有 MEL 权重）  
实验目录：`experiments/13_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELWeightedCE`

## 1. Run12 触发依据

Run12 完成 91 个连续 epoch 后正常早停。5 个保留 bundle 的大小、SHA-256、
fingerprint 与完整状态均通过，auto resume 指向内部 epoch 90；无异常或安全中断。

Run12 相比 Run10 的最佳 image Macro-F1、group Macro-F1、joint 分别提高到
0.55403、0.59969、0.68258，说明 BLG-MDA 注意力改善了整体分类和联合表现；但最低
MEL->NV 仅从 0.32653 降到 0.25510，91 个 epoch 仍无 feasible checkpoint。
Run11 的同一 MEL 权重曾把最低 MEL->NV 降到 0.11224，接近但未越过冻结上限，代价
是整体指标下降。

继续 Run12 不合理：训练已按 patience=20 正常早停，最佳 training fitness 位于内部
epoch 71，完成标记禁止在同目录继续写入。

## 2. 唯一变化因素与 2×2 设计

相对 Run12，唯一变化是将无权重 8 类 CE 替换为 Run11 已冻结的 MEL target weight：

- train-only `n_nv / n_mel = 5539 / 3047 = 1.8178536265178864`；
- 类别顺序 `akiec,bcc,bkl,df,mel,nv,vasc,scc`；
- 权重为 `[1,1,1,1,1.8178536265178864,1,1,1]`；
- image classification loss 总系数仍为 0.5；
- 不使用 val、test 或 external 选择该权重。

该实验补全以下冻结 2×2 因素格：

- Run10：普通 P3/P4 聚合 + unweighted CE；
- Run11：普通 P3/P4 聚合 + MEL-weighted CE；
- Run12：BLG-MDA attention + unweighted CE；
- Run13：BLG-MDA attention + MEL-weighted CE。

因此可同时读取 Run13 相对 Run12 的损失主效应，以及注意力是否缓解 Run11 的整体
指标损失。Run13 仍是相对 Run12 的严格单因素实验。

## 3. 严格保持不变

- Run12 development manifest 逐字节复制，train/val=23,462/1,744；
- source test 634 行排除，YAML `test: null`，external candidates 不进入 manifest；
- fresh `models/yolo26s-sem.pt`，不从 Run12 best/last warm-start；
- YOLO26s backbone、binary semantic main/aux、P3/P4/P5 BLG-MDA 结构与初始化不变；
- detached binary lesion map、identity-initialized ECA、mask-guided mix=0.5 不变；
- partial-label availability、ignore mask、统一 validator、argmax 和 temperature=1 不变；
- standard seeded shuffle，每个 epoch 全部 23,462 train 样本一次；
- batch=16、imgsz=640、epochs=100、patience=20、AdamW、LR、增强、AMP、seed=42；
- 不加入 sampler、focal/asymmetric loss、margin、threshold tuning 或新注意力模块；
- test、ISIC2018 external、PAD-UFES-20 不用于训练、选择或超参数决策。

## 4. 冻结评价和选择

- `joint_score=sqrt(binary_iou * group_macro_f1)`；
- feasible iff group MEL->NV rate <= 0.1077844311 且 MEL support 存在；
- primary alias=`best_feasible_joint`；
- 无 feasible checkpoint 则 Run13 判约束失败，无约束 best 仅作诊断；
- 统一报告 segmentation、image/group 分类、MEL/NV 双向风险、恶性转良性、NLL、
  Brier、ECE、参数量、GFLOPs 和速度；
- 正式模型锁之前禁止运行 test，external 仍只作候选。

## 5. 完整恢复与启动边界

每个完成 epoch 原子发布 `_SUCCESS` bundle，保存 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、stopper、selection、全部 RNG 和 sampler/data progress；
auto/explicit resume 必须校验大小、SHA-256、fingerprint 和完整状态。

只有 Run13 静态/CPU/GPU/真实 development batch 预检全部 PASS、最新 release 文档
明确 `PROJECT_RUN13_GATE=PASS`，并由用户再次确认后，才可从 PyCharm 运行项目根目录
`train_run13_from_pycharm.py`。准备阶段不得调用 test/external evaluator。
