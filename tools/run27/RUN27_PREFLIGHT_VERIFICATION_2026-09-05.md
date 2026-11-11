# Run27 预检复核记录

复核日期：2026-09-05

`RUN27_PREFLIGHT_VERIFICATION=PASS`

`RUN27_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN27_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`FORMAL_TRAINING_STARTED=FALSE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 锁定标识

- 配置 SHA-256：`4732315643841f00107c949bf5585df1c9e07e8255f2813d92da825d892a7619`
- 最终就绪指纹：`85880df0bfb882373d0ffe89fcd2f65b5c3c9bf50ecd9b2f5bf1a4c0a57a10fa`
- Run21 配对基线指纹：`b7b853c4284037a6d51d0e66e10cd0302d167bf4ab30c7b32f512f6f137fb1a9`
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`

## 复核结果

1. Run27 全部 Python 文件语法编译通过。
2. Run27 静态测试及 Run18 延迟早停回归测试共 `8 passed`。
3. CPU 合成两轮断点恢复通过：模型、EMA、优化器、调度器、早停器、选择状态、RNG 和第二轮 batch 顺序与连续训练逐项零差异。
4. RTX 5080 GPU/AMP/workers=2 合成断点恢复通过，上述状态同样逐项零差异。
5. 容量预检通过：batch=16、imgsz=640、AMP、AdamW 更新成功；模型参数 14,507,544，分类器参数 198,920，注意力参数 206，峰值 reserved 显存 7,077,888,000 字节，占设备显存约 41.47%。
6. `--prepare-only` 使用冻结 development manifest 的真实角色 batch 完成：train=23,462、val=1,744，test=null；写出就绪标记，未调用 `trainer.train()`。
7. 最终 v2 正式 Trainer 构造通过：seed=42、epochs=100、patience=30、batch=16、imgsz=640、workers=8、AdamW、AMP、device=0；未创建优化器，未调用 `trainer.train()`。
8. 模型结构复核通过：P3/P4 各一个 identity-initialized ECA 和一个 7×7 恒等初始化空间门，无 P5 注意力、无预测掩码引导，语义分支不变。
9. 正式结果路径 `experiments/27_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed42/results/run27_yolo26m_p3p4_eca_spatial_attention_seed42` 不存在，没有正式训练结果。
10. 未运行 IMA++ test、ISIC 2018 external 或 PAD-UFES-20，未使用任何 test/external 指标，未下载模型或数据。

## 发布结论

用户于 2026-09-05 已明确要求准备到其本人在 PyCharm 点击训练之前，并说明由用户本人点击训练。确认已记录，发布门切换为 PASS。只允许从锁定入口 `tools/run27/train_run27_from_pycharm.py` 启动一次 Run27 正式训练；Codex 未启动正式训练。
