# ab2 预注册协议：固定等权融合消融，seed43

`AB2_PROTOCOL_STATUS=PREREGISTERED`

`POST_TEST1_EXTENSION_CHAIN=TRUE`

`STAGE_D_ADAPTIVE_FUSION_ABLATION=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_AB2_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_AB2_DEVELOPMENT`

## 触发原因

锁定的 gate-balance LGLA 已完成 seed42/43/44。权威 best-Joint 同行三 seed 平均相对 Plain：
Binary IoU `+0.00702`、Image Macro-F1 `+0.01367`、Group Macro-F1 `+0.02501`、Joint
`+0.01700`；三组 seed 的 Image Macro-F1、Group Macro-F1 与 Joint 均同方向提高。根据扩展计划
第 6 节阶段 D，ab2 用于判断收益是否来自自适应全局/局部融合，而不是继续修改完整
LGLA 主结构。

## 唯一预注册改动

ab2 完整复用成功版 lgx seed43 的模型参数图、初始化、P3/P4 尺度、预测病灶局部分支、共享投影、
分类器、数据、增强、优化器、学习率、早停、选择和 checkpoint 合同。唯一改动：

- 完整 LGLA 以学习型两路 softmax 门控逐样本融合 `z_global` 与 `z_local`；
- ab2 固定使用 `z_fused=0.5*z_global+0.5*z_local`；
- 预测病灶注意力仍为 `A=sigmoid(StopGrad(S))`，病灶加权局部描述完全保留；
- 门控 MLP 仍以相同初始化注册在模型/checkpoint 中以保持参数量一致，但 forward 不调用它、无梯度；
- 因不存在学习型门控 logits，gate-balance 正则同步禁用，权重锁定为 `0.0`。

ab2 不使用真实 mask 作为分类输入，不新增 P5、GeM、ROIAlign、独立局部分支投影、学习型空间注意力、
类别权重、风险损失、label smoothing、采样器、增强或阈值。固定等权的目的不是提出新模型，而是只
删除“根据样本自适应改变全局/局部占比”的能力。

## 冻结训练合同

- seed=43，fresh start=`models/yolo26m-sem.pt`，与完整 LGLA lgx、Plain Run22 同 seed 配对；
- development train/val=`23,462/1,744`，test、external、PAD-UFES-20 禁止进入开发；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- CE 无类别权重、无 label smoothing，image classification weight=`0.5`；
- gate-balance weight=`0.0`，delayed utility-gated early stop、Joint、风险约束、checkpoint v2
  与 alias 全部保持不变；
- 正式输出独占 `experiments/ab2_YOLO26m_Binary8Class_LGLAv1_FixedEqualFusion_Seed43`；
- 唯一入口默认 `resume=auto`；用户点击前正式结果目录必须不存在。

## 比较与解释规则

只比较各自 `best_joint_score` alias 对应的同一 CSV 行：

1. 主比较为完整 LGLA lgx seed43 减 ab2 seed43；
2. Plain Run22 仅作为共同参照；
3. 若完整 LGLA 的 Group Macro-F1 与 Joint 均高于 ab2，则支持自适应门控对核心收益有贡献；
4. 若 ab2 持平或更好，则不能把完整 LGLA 的提升归因于自适应门控；
5. 无论结果如何，都不根据 ab2 validation 继续调结构；与 ab1 一起审计后进入统一汇总。

## 启动边界

唯一正式入口：`tools/ab2/train_ab2_from_pycharm.py`。AI 可执行静态、数据边界、GPU、checkpoint
save/resume、容量、prepare-only 和 Trainer 构造预检，但不得调用正式训练路径。长训练必须由用户
亲自在 PyCharm 点击。
