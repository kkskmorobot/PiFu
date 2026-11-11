# ab1 预注册协议：去除病灶引导消融，seed43

`AB1_PROTOCOL_STATUS=PREREGISTERED`

`POST_TEST1_EXTENSION_CHAIN=TRUE`

`STAGE_D_LESION_GUIDANCE_ABLATION=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_AB1_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_AB1_DEVELOPMENT`

## 触发原因

锁定的 gate-balance LGLA 已完成 seed42/43/44。权威 best-Joint 同行三 seed 平均相对 Plain：
Binary IoU `+0.00702`、Image Macro-F1 `+0.01367`、Group Macro-F1 `+0.02501`、Joint
`+0.01700`；三组 seed 的 Image Macro-F1、Group Macro-F1 与 Joint 均同方向提高。根据扩展计划
第 6 节阶段 D，下一项优先实验为 ab1，用于判断收益是否来自病灶空间先验，而不是继续修改完整
LGLA 主结构。

## 唯一预注册改动

ab1 完整复用成功版 lgx seed43 的模型参数图、初始化、P3/P4 尺度、共享投影、双分支、两路 softmax
门控、门控平衡正则、分类器、数据、增强、优化器、学习率、早停、选择和 checkpoint 合同。唯一改动：

- 完整 LGLA 的局部分支权重为 `A=sigmoid(StopGrad(S))`；
- ab1 把 `A` 固定替换为全图常数 1；
- 因而 `z_local=sum(F*1)/sum(1)=z_global`；
- 仍保留与完整 LGLA 相同的共享投影、门控和参数量，形成参数匹配的无病灶引导反事实。

ab1 不使用真实 mask 作为分类输入，不新增 P5、GeM、ROIAlign、独立局部分支投影、学习型空间注意力、
类别权重、风险损失、label smoothing、采样器、增强或阈值。常数权重的目的不是提出新模型，而是只
删除“预测病灶位置与特征汇聚的对应关系”。

## 冻结训练合同

- seed=43，fresh start=`models/yolo26m-sem.pt`，与完整 LGLA lgx、Plain Run22 同 seed 配对；
- development train/val=`23,462/1,744`，test、external、PAD-UFES-20 禁止进入开发；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- CE 无类别权重、无 label smoothing，image classification weight=`0.5`；
- gate-balance weight=`0.01`，delayed utility-gated early stop、Joint、风险约束、checkpoint v2
  与 alias 全部保持不变；
- 正式输出独占 `experiments/ab1_YOLO26m_Binary8Class_LGLAv1_Unguided_GateBalance_Seed43`；
- 唯一入口默认 `resume=auto`；用户点击前正式结果目录必须不存在。

## 比较与解释规则

只比较各自 `best_joint_score` alias 对应的同一 CSV 行：

1. 主比较为完整 LGLA lgx seed43 减 ab1 seed43；
2. Plain Run22 仅作为共同参照；
3. 若完整 LGLA 的 Group Macro-F1 与 Joint 均高于 ab1，则支持病灶引导对核心收益有贡献；
4. 若 ab1 持平或更好，则不能把完整 LGLA 的提升归因于病灶空间先验；
5. 无论结果如何，都不根据 ab1 validation 继续调结构；先回到用户，再决定是否按原计划准备 ab2。

## 启动边界

唯一正式入口：`tools/ab1/train_ab1_from_pycharm.py`。AI 可执行静态、数据边界、GPU、checkpoint
save/resume、容量、prepare-only 和 Trainer 构造预检，但不得调用正式训练路径。长训练必须由用户
亲自在 PyCharm 点击。
