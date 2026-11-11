# Run27 预注册协议：P3/P4 ECA + 恒等空间注意力

`RUN27_PROTOCOL_LOCKED=TRUE`

`TEST_POLICY=FORBIDDEN_UNTIL_FINAL_MODEL_LOCK_AND_FRESH_USER_AUTHORIZATION`

`EXTERNAL_POLICY=FORBIDDEN_FOR_TRAINING_SELECTION_AND_THRESHOLD_TUNING`

## 研究问题与配对依据

三组 plain/ECA 配对显示纯通道注意力没有稳定提高分类或联合指标。Run26 在 seed44 上增加空间门后，相对同 seed 的 Run25 ECA-only，Joint +0.01007、Group Macro-F1 +0.01391、Image Macro-F1 +0.02572、Binary IoU +0.00530、MEL→NV -0.11225。Run27 预注册为 seed42 的严格配对复现，用于检验这项单 seed 改善能否跨随机种子保持。

Run27 的直接因果对照是 Run21（seed42、P3/P4 ECA-only）；同 seed plain 参照是 Run18。Run26 只提供跨 seed 的先验结果，不能替代 Run21 的配对基线。

## 唯一预注册因素

相对 Run21，只在每个 identity-initialized ECA 输出后增加一个空间门：

1. 对特征沿通道计算 mean 与 max；
2. 拼接成 2 通道空间描述；
3. 通过无 bias 的 7×7 卷积；
4. 使用 `2*sigmoid` 得到乘法门；
5. 卷积权重全零初始化，因此初始门严格为 1。

空间注意力仅服务图像分类头；原始 P3/P4 仍直接进入语义分割头。禁止预测掩码引导、P5 描述符或反向修改语义分支。

## 冻结训练合同

- seed=42，与 Run21 严格配对；
- 初始化 `models/yolo26m-sem.pt`，AMP 校验仅使用本地 `tools/yolo26n.pt`，禁止运行时下载；
- development train/val 与 Run21 完全相同；test=null，外部候选不得进入训练、选择或调参；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- partial-label loss、unweighted CE、image classification weight=0.5、数据采样和增强均不变；
- 延迟早停从 Trainer epoch 40 开始计数，patience=30，最早停止 Trainer epoch 70；
- Binary IoU≥0.70、Group Macro-F1≥0.50、mel→nv≤0.1077844311，主选择别名仍为 `best_feasible_joint`；
- 无风险可行点时仅报告 constraint failure 与 best joint 诊断点。

## 启动与后续

唯一正式入口为 `tools/run27/train_run27_from_pycharm.py`，默认 `resume=auto`。首次启动前正式实验目录必须不存在，且 release 文档必须记录用户确认并为 PASS。

Run27 完成后，先以各 run 的 `best_joint_score` 同一 CSV 行与 Run21 计算 `Run27−Run21`，并另报告相对 Run18 plain 的总效果；bundle epoch 为零基，CSV epoch 为一基，必须按 `csv_epoch=bundle_epoch+1` 对齐。不得拼接不同 epoch 的独立峰值充当一个 checkpoint。

Run27 单独完成后不锁定注意力主线，继续等待预注册的 seed43 Run28。三 seed 的稳定收益判据预先固定为：ECA+spatial 相对 ECA-only 的 Joint 与 Group Macro-F1 配对均值为正、各 seed 方向大体一致，并且 MEL→NV 不出现系统性恶化。若不满足，则结束这一注意力架构方向；不得训练后修改判据。正式模型锁定前禁止 test 和外部评价。
