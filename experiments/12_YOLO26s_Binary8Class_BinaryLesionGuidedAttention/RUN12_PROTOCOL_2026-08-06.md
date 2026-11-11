# PiFuV2 Run12 预注册协议

日期：2026-08-06  
主基线：`experiments/10_YOLO26s_Binary8Class_PartialLabel_Baseline`  
诊断对照：`experiments/11_YOLO26s_Binary8Class_MELWeightedCE`  
实验目录：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`

## 1. 触发依据

Run10 的最佳 group Macro-F1 / joint 为 0.59426 / 0.68132，但 100 个 epoch
没有 feasible checkpoint，最佳 joint 状态的 MEL->NV 为 0.43878。Run11 仅把
MEL target CE 权重提高到 1.8178536265，最低 MEL->NV 从 0.32653 降到
0.11224，但仍高于 0.1077844311；同时最佳 group Macro-F1 / joint 下降到
0.57017 / 0.66254。直接损失重加权改善了风险方向，却牺牲了整体分类、联合分数
与校准，不能建立正式模型锁。

本项目研究主题是基于注意力机制改进 YOLO。新二值分割头已经提供比旧 Run7
八通道疾病语义更直接的病灶概率，因此 Run12 在新数据合同上重新检验病灶引导
注意力，而不是继续提高类别损失权重。

## 2. 唯一变化因素

相对 Run10，只替换图像分类特征聚合：

- 不变的二值语义主头产生 `sigmoid(lesion_logit)` soft lesion map；
- soft lesion map 在分类分支 detach，不增加分类到分割头的第二条梯度路径；
- P3/8、P4/16、P5/32 各通过一个 identity-initialized ECA 通道门；
- 每尺度描述子固定为 50% global average + 50% normalized lesion-guided average；
- 三尺度描述子拼接后进入与 Run10 同 hidden width、SiLU、dropout=0.20、8-logit
  的 MLP；
- P3/P4 分类器初始化逐元素继承 Run10 构造，新增 P5 输入块初始贡献为 0。

该完整聚合模块作为一个预注册架构因素，简称 BLG-MDA
（Binary-Lesion-Guided Multi-scale Dual Attention）。

## 3. 严格保持不变

- Run10 development manifest 的 train/val 行逐字节复制，数据投影和 split 不变；
- train/val=23,462/1,744，partial-label availability 和 ignore-mask 语义不变；
- `models/yolo26s-sem.pt` fresh start、YOLO26s backbone、二值语义主/辅助头不变；
- unweighted 8-class CE、image classification loss weight=0.5、label smoothing=0；
- 标准 seeded shuffle，无 class/lesion-balanced sampler，每轮全部 train 样本一次；
- batch=16、imgsz=640、100 epochs、patience=20、AdamW、LR、增强、AMP、seed=42；
- binary loss、partial-label mask、validator、风险阈值、选择和断点合同不变；
- 不加入 Run11 MEL 权重、CBAM、focal/asymmetric loss、margin、temperature fitting
  或 threshold tuning；
- test、ISIC2018 external、PAD external 不进入训练、选择或超参数决策。

## 4. 研究假设与判定

主要假设：显式病灶空间聚合能够减少背景/设备线索，使 MEL 与 NV 的判别更多依赖
病灶颜色、纹理和边界，在不使用类别代价重加权的情况下改善 MEL->NV。

冻结选择：

- `joint_score=sqrt(binary_iou * group_macro_f1)`；
- feasible iff group MEL->NV rate <= 0.1077844311 且 MEL support 存在；
- primary alias=`best_feasible_joint`；
- 若没有 feasible checkpoint，Run12 仍判约束失败；无约束 best 只作诊断；
- 必须同时报告分割、image/group 分类、双向 MEL/NV 风险、恶性转良性、NLL、
  Brier、ECE、参数量、GFLOPs 和速度；正式模型锁前不运行 test。

## 5. 完整恢复与启动边界

每个完成 epoch 原子发布 `_SUCCESS` bundle，恢复 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、stopper、selection、全部 RNG、sampler/data progress、manifest
哈希和 behavior fingerprint。auto/explicit resume 拒绝不完整或指纹不一致 bundle。

正式训练只能在全部 Run12 门禁通过、最新 release 文档写出
`PROJECT_RUN12_GATE=PASS` 且用户再次确认后，从 PyCharm 运行项目根目录
`train_run12_from_pycharm.py`。准备和冒烟阶段不得调用 test/external evaluator。
