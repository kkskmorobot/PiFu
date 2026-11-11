# Run33 预注册协议：YOLO26m P3/P4 标准 CBAM seed43

`RUN33_PROTOCOL_STATUS=PREREGISTERED`

`TEST_POLICY=FORBIDDEN_UNTIL_FINAL_MODEL_LOCK_AND_FRESH_USER_AUTHORIZATION`

`EXTERNAL_POLICY=FORBIDDEN_FOR_TRAINING_SELECTION_AND_THRESHOLD_TUNING`

## 研究问题与基线

Run32 未通过风险辅助头复现门，训练期风险损失/辅助头路线停止。依据根目录后续路线图，Run33
用于补齐 YOLO26m 正式体系中的标准 CBAM 对照。直接基线为同 seed43 的 Run22 plain P3/P4
分类头；Run33 不继承 Run29～Run32 的任何风险损失或辅助头。

## 唯一预注册因素

相对 Run22，只在图像分类头使用的 P3、P4 特征上各增加一个标准 CBAM：

1. 通道注意力采用全局平均池化与全局最大池化，共享两层 MLP，reduction=16；
2. 通道门使用标准 sigmoid；
3. 空间注意力采用通道 mean/max 描述符和无 bias 的 7×7 卷积；
4. 空间门使用标准 sigmoid；
5. 按“通道后空间”的标准顺序执行，然后沿用 Run22 的 global-average pooling 与 MLP；
6. 使用 PyTorch 常规参数初始化，不使用 ECA、`2*sigmoid` 或恒等初始化。

注意力只服务图像分类头；原始 P3/P4 仍直接进入语义分割头。禁止预测 mask 引导、P5 描述符、
BLG-MDA、风险辅助头或反向修改语义分支。因此本实验是当前统一 YOLO26m 底座上的标准 CBAM
对照，并非对 Run6 插入位置和旧七分类任务的逐字复制。

## 冻结训练合同

- seed=43，与 Run22 严格配对；
- 初始化 `models/yolo26m-sem.pt`；
- development train/val、冻结 manifest/split/mask/taxonomy 与 Run22 完全一致；
- binary segmentation + official 8-class partial-label，多任务 image weight=0.5；
- unweighted CE、标准 epoch shuffle、固定增广、batch=16、imgsz=640、workers=8、AdamW、AMP；
- epochs=100，Trainer epoch 40 后启用 patience=30 的 utility-gated early stopping；
- validation 指标、Joint、风险约束、选择别名、checkpoint v2 和恢复合同保持 Run22 不变；
- `test: null`，禁止 IMA++ test、ISIC2018 external 和 PAD-UFES-20。

## seed43 后的预注册判定门

比较 Run33 与 Run22 各自 `best_joint_score` bundle 对应的同一 CSV 行，不拼接独立峰值。

必须同时满足效用护栏：

- ΔBinary IoU ≥ -0.005；
- ΔGroup Macro-F1 ≥ -0.010；
- ΔJoint ≥ -0.005。

并至少满足一项实质收益：

- ΔImage Macro-F1 ≥ +0.010；或
- ΔGroup Macro-F1 ≥ +0.010；或
- ΔJoint ≥ +0.005；或
- ΔMEL→NV ≤ -0.030；或
- Δmalignant→benign ≤ -0.020。

满足门限才预注册 seed42、seed44 的完全同构复现；否则标准 CBAM 在 seed43 后停止，作为负向或
有限对照写入论文，并进入统一模型锁定。不得训练后修改门限或改为恒等初始化追逐 validation。

## 启动边界

唯一正式入口为 `tools/run33/train_run33_from_pycharm.py`，默认 `resume=auto`。首次启动前正式结果
目录必须不存在，静态/GPU/恢复预检和发布门必须全部 PASS。Codex 只准备到入口展示，必须由用户
在 PyCharm 点击 Run；正式模型锁定前禁止 test 和 external。

