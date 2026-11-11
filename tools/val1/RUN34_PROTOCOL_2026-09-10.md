# Run34 协议：Plain YOLO26m 三 seed 固定代价决策

`RUN34_PROTOCOL_STATUS=PREREGISTERED`

## 定位

Run34 不是训练 run。它是模型架构锁定后的 development-only 决策规则实验，使用 Run18、Run22、Run24 已冻结的 Plain YOLO26m best-Joint bundle。

规范入口为 `evaluate_run34_from_pycharm.py`。为了不破坏用户已建立的 PyCharm 运行配置，`run_run34_from_pycharm.py` 仅作为兼容转发器保留。

评估时先从 `trainer.pt` 恢复模型结构，再用同一 bundle 的 `runtime_state.pt::training_state.ema` 加载无损 fp32 EMA。不得直接用 `trainer.pt` 中用于压缩存储的 fp16 EMA 作为正式决策评估权重。

验证调用须保持训练期原路径：未融合 Conv+BN 的 fp32 EMA 权重，CUDA autocast 推理，全程无梯度。不得改用 standalone AutoBackend 的自动 fuse/纯 fp32 路径。

## 单一因素

模型、权重、输入、分组和概率聚合全部不变。唯一变化是将八类 argmax 换成预注册的后验期望代价最小化：

- 正确分类代价 0；
- 普通误分代价 1；
- 真实恶性类 `bcc/mel/scc` 被判为良性类的代价 4；
- 真实 `mel` 被判为 `nv` 的代价 8；
- 决策为使后验期望代价最小的类别，平局时取较小 class id。

不允许在看到 Run34 结果后修改代价并在同一 validation 上重试。

## 接受门

相对各 seed 原始 argmax，须同时满足：

1. 每个 seed 的 MEL→NV 与 malignant→benign 均不增加；
2. 三 seed 平均 MEL→NV 绝对降幅至少 0.10；
3. 三 seed 平均 malignant→benign 绝对降幅至少 0.05；
4. 平均 Group Macro-F1 降幅不超过 0.03；
5. 平均 Joint 降幅不超过 0.02；
6. 任一 seed 的 Group Macro-F1 降幅不超过 0.06。

通过时锁定固定代价规则；失败时保留 argmax，结束决策规则调整，不继续追逐 validation。

## 边界

- 只允许读取 Run22 的 development-only YAML/manifest 与三个冻结 bundle。
- 入口不提供 test/external 参数。
- 不创建 optimizer，不调用 backward，不修改模型权重。
- IMA++ test、ISIC2018 external 和 PAD-UFES-20 仍禁止。
