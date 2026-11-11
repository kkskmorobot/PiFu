# lg1 预注册协议：LGLA v1 seed43 首轮筛选

`LG1_PROTOCOL_STATUS=PREREGISTERED`

`POST_TEST1_EXTENSION_CHAIN=TRUE`

`TEST_POLICY=FORBIDDEN_DURING_LG1_DEVELOPMENT`

`EXTERNAL_POLICY=FORBIDDEN_DURING_LG1_DEVELOPMENT`

## 研究问题与严格基线

lg1 检验 LGLA v1（Lesion-Guided Global–Local Attention）能否在冻结的 Plain YOLO26m
partial-label 多任务底座上提高八类图像分类与综合效用。直接配对基线为 Run22 seed43；Run18
seed42 与 Run24 seed44 只作为后续通过首轮门后可准备的配对基线。

lg1 属于用户于 2026-09-12 授权的独立 Post-Test1 本科论文扩展链。它不改变、不覆盖也不重新解释
Run1～Run33、Val1、Sum1 或 Test1。lg1 阶段只允许 development train/val，不读取 IMA++ test、
ISIC2018 incremental external 或 PAD-UFES-20。

## 唯一预注册因素：LGLA v1 分类头

相对 Run22，只把 Plain P3/P4 图像分类路径替换为以下 LGLA v1：

1. 原语义头先从 P3 产生单通道主分割 logit `S`；
2. 分类注意力严格使用 `A=sigmoid(StopGrad(S))`，训练和推理均不使用真实 mask；
3. `A` 以双线性插值缩放到 P3/P4，`align_corners=false`；
4. 每个尺度同时计算全局平均描述符与病灶概率加权描述符；
5. P3/P4 描述符分别以同尺度共享线性层投影到 128 维，再拼接为全局、局部各 256 维；
6. `MLP([z_global,z_local])` 输出二路 softmax 权重，融合后进入 256 隐层、dropout=0.20 的八分类器；
7. 门控末层以零权重/零 bias 初始化，所以初始权重严格为 0.5/0.5；
8. 加权池化使用 `eps=1e-6`；注意力质量低于空间面积的 `1e-4` 时，局部分支逐样本回退到全局池化；
9. 门控权重、各尺度注意力质量与回退标志保留为 detached 诊断张量，可在后续锁定验证中记录。

语义分割头仍读取原始 P3/P4；分类损失不会经注意力概率图反传到语义 classifier。共享主干接收分类
梯度的方式与 Run22 相同。lg1 不加入 P5、ROIAlign、GeM、风险损失、类别权重、label smoothing、
新采样器、新增强或新阈值。

## 冻结训练合同

- seed=43，与 Run22 配对；fresh start=`models/yolo26m-sem.pt`；
- development train/val=`23,462/1,744`，分割监督=`10,483/292`，分类监督=`13,715/1,488`；
- classification-only mask 的 255 只作 ignore，不作为病灶注意力真值；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- 优化器、学习率策略、增强、image classification weight=0.5、unweighted CE 全部保持 Run22；
- Trainer epoch 40 后启用 patience=30，最早停止 epoch 70；
- validation、Joint、风险约束、checkpoint v2、专项 alias 和完整恢复合同保持 Run22；
- 正式结果目录必须在用户点击前不存在；唯一入口默认 `resume=auto`。

## 首轮筛选门

比较 lg1 与 Run22 各自 `best_joint_score` bundle 对应的同一 CSV 行，不拼接独立峰值。建议进入
多 seed 复现必须同时满足：

- Group Macro-F1 差值不低于 `+0.005`；
- Joint 差值不低于 `+0.003`；
- Image Macro-F1 不下降；
- Binary IoU 差值不低于 `-0.005`；
- 注意力、门控、损失与恢复状态全部有限；冻结 val 上局部门均值位于 `[0.05,0.95]`，标准差不低于 `0.001`。

任一项略低但分类主指标有明显可解释提升时，只报告完整差值与权衡，由用户决定是否继续；不得自行
放宽门限。明确失败则停止，不准备 lg2/lg3，只允许在诊断支持且用户确认后预注册一次 lgx。

## 启动边界

唯一正式入口：`tools/lg1/train_lg1_from_pycharm.py`。AI 可执行静态、CPU、GPU、save/resume、
prepare-only 和 Trainer 构造预检，但不得调用该入口的正式训练路径，不得替用户启动长训练。

