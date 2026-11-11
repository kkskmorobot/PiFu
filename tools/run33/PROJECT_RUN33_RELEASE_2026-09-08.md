# Run33 正式发布（2026-09-08）

`PROJECT_RUN33_GATE=PASS`

`RUN33_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 发布结论

Run32 未通过风险辅助头复现门，已停止训练期风险损失/辅助头路线。依据根目录
`PROJECT_POST_RUN32_EXPERIMENT_PLAN_2026-09-08.md`，Run33 发布为同 seed43 的 YOLO26m P3/P4
标准 CBAM 单因素对照，直接基线为 Run22。

## 唯一因素

相对 Run22，只在分类头使用的 P3、P4 特征上各增加一个常规标准 CBAM：共享 avg/max channel
MLP（reduction=16）后接 channel mean/max 的 7×7 spatial gate，均使用标准 sigmoid 和 PyTorch
常规初始化。原始 P3/P4 继续直接进入语义分割头；无 ECA、无 P5、无 mask 引导、无风险辅助头。

## 锁定发布对象

- 唯一入口：`tools/run33/train_run33_from_pycharm.py`
- 配置 SHA-256：`9dc2ab8135ed32709998d0ed086400d098719d982be63a2ee623b527b57c9ae7`
- fingerprint：`95d6e3a2fb3bb5aa05ce36ababead5f9f872272a1890e7bca03bdc96c94c533e`
- 正式结果目录：
  `experiments/33_YOLO26m_Binary8Class_P3P4_StandardCBAM_Seed43/results/run33_yolo26m_p3p4_standard_cbam_seed43`

## 发布门

- 静态编译与 6/6 单元测试：PASS；
- development-only `--prepare-only`：PASS；
- 正式 Trainer 构造：PASS；
- RTX 5080 batch16/img640/AMP 容量：PASS；
- 合成中断/恢复精确一致性：PASS；
- 正式结果目录不存在，`trainer.train()` 未调用；
- test/external/PAD 均未使用。

## 启动边界

用户本轮已明确要求依据 Run32 结果选择并准备下一次训练。Codex 只能把锁定入口展示到 PyCharm，
不得点击 Run；正式启动必须由用户本人在 PyCharm 点击。入口默认 `resume=auto`，首次运行将 fresh
start。训练完成后按预注册门与 Run22 的 best-Joint 同行比较，再决定是否复现 seed42/44。

