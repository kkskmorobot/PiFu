# Run28 预注册协议：seed43 P3/P4 ECA + 恒等空间注意力复现

`RUN28_PROTOCOL_LOCKED=TRUE`

`TEST_POLICY=FORBIDDEN_UNTIL_FINAL_MODEL_LOCK_AND_FRESH_USER_AUTHORIZATION`

`EXTERNAL_POLICY=FORBIDDEN_FOR_TRAINING_SELECTION_AND_THRESHOLD_TUNING`

## 研究问题与配对依据

Run26 在 seed44 上显示空间门相对 ECA-only 的正向收益；Run27 在 seed42 上则显示 Joint、Group Macro-F1、Image Macro-F1、Binary IoU 和 Group Recall 均下降，仅 MEL→NV 略有改善。两组方向冲突，不能根据单个 seed 锁定或临时修改架构。

Run28 是预先规划的第三组 seed43 严格配对复现。直接因果对照为 Run23（seed43、P3/P4 ECA-only），同 seed plain 参照为 Run22。Run28 用于完成三 seed 判定，不是根据 Run27 结果设计的新结构。

## 唯一预注册因素

相对 Run23，只在每个 identity-initialized ECA 输出后增加一个空间门：

1. 对特征沿通道计算 mean 与 max；
2. 拼接为 2 通道空间描述；
3. 通过无 bias 的 7×7 卷积；
4. 使用 `2*sigmoid` 得到乘法门；
5. 卷积权重全零初始化，因此初始门严格为 1。

空间注意力仅服务图像分类头；原始 P3/P4 仍直接进入语义分割头。禁止预测掩码引导、P5 描述符或反向修改语义分支。

## 冻结训练合同

- seed=43，与 Run23 严格配对；
- 初始化 `models/yolo26m-sem.pt`，AMP 校验只使用本地 `tools/yolo26n.pt`，禁止运行时下载；
- development train/val 与 Run23 完全相同；`test=null`，外部候选不得进入训练、选择或调参；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- partial-label loss、unweighted CE、image classification weight=0.5、采样和增强不变；
- 延迟早停从 Trainer epoch 40 开始计数，patience=30，最早停止 Trainer epoch 70；
- Binary IoU≥0.70、Group Macro-F1≥0.50、MEL→NV≤0.1077844311，主选择别名仍为 `best_feasible_joint`；
- 无风险可行点时只报告 constraint failure 与 `best_joint_score` 诊断点。

## 预注册分析规则

Run28 完成后，以各 run 的 `best_joint_score` 同一 CSV 行计算三组 `ECA+spatial − ECA-only` 配对差值：seed42 为 Run27−Run21，seed43 为 Run28−Run23，seed44 为 Run26−Run25。另分别报告相对同 seed plain 的总效果。bundle epoch 为零基、CSV epoch 为一基，必须按 `csv_epoch=bundle_epoch+1` 对齐，不得拼接不同 epoch 的独立峰值。

空间注意力“稳定收益”判据固定为：Joint 与 Group Macro-F1 的三 seed 配对均值为正、各 seed 方向大体一致，并且 MEL→NV 不出现系统性恶化。若不满足，则停止当前注意力架构主线并如实登记负向消融；不得训练后修改判据，也不得通过叠加 P5、预测 mask 或额外门追逐 development val。

## 启动边界

唯一正式入口为 `tools/run28/train_run28_from_pycharm.py`，默认 `resume=auto`。首次启动前正式实验目录必须不存在，全部预检和发布门必须 PASS，并由用户本人从 PyCharm 点击启动。正式模型锁定前禁止 test 和外部评价。
