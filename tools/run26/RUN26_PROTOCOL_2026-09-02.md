# Run26 预注册协议：P3/P4 ECA + 恒等空间注意力

`RUN26_PROTOCOL_LOCKED=TRUE`

`TEST_POLICY=FORBIDDEN_UNTIL_FINAL_MODEL_LOCK_AND_FRESH_USER_AUTHORIZATION`

`EXTERNAL_POLICY=FORBIDDEN_FOR_TRAINING_SELECTION_AND_THRESHOLD_TUNING`

## 研究问题

三组 plain/ECA 配对显示纯通道注意力没有稳定提高分类或联合指标。Run26 检验在相同 P3/P4 分类特征上补充轻量空间注意力，能否改善皮肤病识别、Group Macro-F1 与联合表现，同时保持二值分割质量。

## 唯一预注册因素

相对 Run25，只在每个 identity-initialized ECA 输出后增加一个空间门：

1. 对特征沿通道计算 mean 与 max；
2. 拼接成 2 通道空间描述；
3. 通过无 bias 的 7×7 卷积；
4. 使用 `2*sigmoid` 得到乘法门；
5. 卷积权重全零初始化，因此初始门严格为 1。

空间注意力仅服务图像分类头；原始 P3/P4 仍直接进入语义分割头。禁止预测掩码引导、P5 描述符或反向修改语义分支。

## 冻结训练合同

- seed=44，与 Run25 严格配对；
- 初始化 `models/yolo26m-sem.pt`，AMP 校验仅使用本地 `tools/yolo26n.pt`，禁止运行时下载；
- development train/val 与 Run25 完全相同；test=null，外部候选不得进入训练、选择或调参；
- epochs=100、batch=16、imgsz=640、workers=8、AdamW、AMP、确定性训练；
- partial-label loss、unweighted CE、image classification weight=0.5、数据采样和增强均不变；
- 延迟早停从 Trainer epoch 40 开始计数，patience=30，最早停止 Trainer epoch 70；
- Binary IoU≥0.70、Group Macro-F1≥0.50、mel→nv≤0.1077844311，主选择别名仍为 `best_feasible_joint`；
- 无风险可行点时仅报告 constraint failure 与 best joint 诊断点。

## 启动与后续

唯一正式入口为 `tools/run26/train_run26_from_pycharm.py`，默认 `resume=auto`。首次启动前正式实验目录必须不存在，且 release 文档必须记录用户确认并为 PASS。

Run26 完成后先与同 seed 的 Run25、Run24 比较。如果空间门相对 Run25 改善 joint、Group Macro-F1 且不显著恶化 mel→nv，再进入多 seed 复现；否则停止该注意力方向。正式模型锁定前禁止 test 和外部评价。
