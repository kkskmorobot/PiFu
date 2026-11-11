# Run21 预注册协议：P3/P4 ECA-only 通道注意力

`RUN21_PROTOCOL_STATUS=PREREGISTERED`

## 唯一预注册因素

相对 Run18 plain 基线，只在分类头原有的 P3、P4 特征上各增加一个 kernel size 5 的 ECA 通道门，再执行与 Run18 相同的 global-average pooling 和 MLP：

- ECA 卷积权重全零初始化，使初始乘法门严格等于 1；
- 只由分类损失学习两个 ECA 模块；
- 不使用分割概率图、病灶空间池化或 P5 描述符；
- P3/P4 原始特征仍直接进入语义分割头，语义分支不修改。

数据、split、manifest、partial-label loader、损失、采样、增广、YOLO26m 权重、batch=16、imgsz=640、AdamW、seed=42、100 epochs、延迟早停、统一 val、joint-score-2、风险约束与 checkpoint v2 合同全部保持 Run18 不变。

## 数据和评价边界

只允许冻结 development train/val。IMA++ test、ISIC 2018 Task 1 外部候选与 PAD-UFES-20 均不得用于训练、选择、阈值拟合或本阶段评价。正式模型锁定并取得新授权前禁止 test 和外部评价。

## 训练入口

唯一入口：`tools/run21/train_run21_from_pycharm.py`。默认 `resume=auto`；最早可见 epoch 70 早停，最多 100 epochs。

## 判断标准

Run21 首先与 Run18 比较 ECA-only 的净增益；Run19/Run20 用于解释空间病灶引导和 P5 组合的影响。没有 feasible checkpoint 时仍只报告约束失败，不在本轮混入损失、采样或风险阈值改动。
