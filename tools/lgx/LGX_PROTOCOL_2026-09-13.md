# lgx 预注册协议：LGLA v1 门控平衡修订，seed43

`LGX_PROTOCOL_STATUS=PREREGISTERED`

`POST_TEST1_EXTENSION_CHAIN=TRUE`

`SINGLE_ALLOWED_REVISION_AFTER_LG1_FAILURE=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_LGX_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_LGX_DEVELOPMENT`

## 触发原因

lg1 已完整训练 100 个 epoch，但未通过首轮筛选门。权威 best-Joint 同行（CSV epoch 84、bundle
`epoch_0083`）相对 Run22 的 Group Macro-F1、Image Macro-F1 与 Joint 均下降。冻结 validation
诊断进一步显示局部门权均值约 `8.60e-13`、总体标准差约 `3.41e-12`，全局门权为 1，证明自适应
门控已系统性塌缩；注意力质量有限、无 NaN，且没有空注意力回退。因此按扩展计划第 6 节阶段 C2，
不直接准备 lg2/lg3，而只允许一次 lgx 门控修订。

## 唯一预注册改动

lgx 完整复用 lg1 的 P3/P4 LGLA v1、初始化、数据、增强、优化器、学习率、早停、选择和 checkpoint
合同，只增加一个作用于有官方分类标签样本的批次均值门控-logit平衡正则。

设两路门控 softmax 之前的 logits 为 `g_global` 与 `g_local`，当前批次有分类标签的样本集合为
`C`：

`d_i = g_local_i - g_global_i`

`L_gate_balance = (mean_{i in C}(d_i))^2`

`L_total = L_semantic + image_cls_weight * batch_size * (CE + 0.01 * L_gate_balance)`

系数固定为 `0.01`。该正则只阻止整个有标签批次系统性偏向同一路，不惩罚各样本围绕平衡均值的
差异，因此仍允许自适应门控学习样本差异。无分类标签样本不参与该正则。推理结构和推理公式不变。

lgx 不加入 P5、GeM、ROIAlign、温度、LayerNorm、门控权重硬裁剪、熵正则、类别权重、风险损失、
label smoothing、采样器、新增强或新阈值；不得把多个修正的收益归因于本次单因素实验。

## 冻结训练合同

- seed=43，fresh start=`models/yolo26m-sem.pt`，与 Run22、lg1 配对；
- development train/val=`23,462/1,744`，测试集、外部集和 PAD-UFES-20 不进入开发；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- 原 CE 仍无类别权重且无 label smoothing，image classification weight=`0.5`；
- 原 delayed utility-gated early stop、validation、Joint、风险约束、checkpoint v2 与 alias 不变；
- 正式输出独占 `experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43`；
- 唯一入口默认 `resume=auto`，用户点击前正式结果目录必须不存在。

## 筛选与停止规则

仍以 lgx 与 Run22 各自 `best_joint_score` bundle 的同一 CSV 行比较，不拼接独立峰值。通过必须同时满足：

- Group Macro-F1 差值不低于 `+0.005`；
- Joint 差值不低于 `+0.003`；
- Image Macro-F1 不下降；
- Binary IoU 差值不低于 `-0.005`；
- 冻结 validation 局部门均值位于 `[0.05,0.95]`、标准差不低于 `0.001`，且全部诊断有限。

lgx 通过后也必须先把完整结果交给用户，由用户决定是否补 seed42/44 或消融。lgx 若未通过，则按原
计划停止 LGLA 继续训练路线，保留 lg1/lgx 为负向或定向风险改善实验，不继续在 validation 上堆模块。

## 启动边界

唯一正式入口：`tools/lgx/train_lgx_from_pycharm.py`。AI 可执行静态、数据边界、GPU、checkpoint
save/resume、prepare-only 和 Trainer 构造预检，但不得调用正式训练路径。长训练必须由用户亲自在
PyCharm 点击。

