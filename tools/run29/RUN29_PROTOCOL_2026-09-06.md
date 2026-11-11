# Run29 预注册协议：plain seed43 + 恶性组层级辅助损失

`RUN29_PROTOCOL_STATUS=PREREGISTERED`

## 决策依据

Run26/27/28 的 P3/P4 ECA+spatial7 相对同 seed ECA-only 的 Joint 改变量分别为
`+0.01007 / -0.02993 / +0.02032`，三 seed 平均仅 `+0.00015`；Group Macro-F1
平均仅 `+0.00226`，且 seed42 明显反向。虽然 MEL→NV 三个 seed 均改善，空间门没有形成
稳定的整体收益，因此不锁定注意力主线。

Run29 转入预先规划的独立风险损失实验，以 Run22 的 plain seed43 为严格配对基线。

## 唯一预注册因素

仅在 Run22 原有 8 类无权重交叉熵上增加权重为 `0.5` 的恶性组辅助 NLL：

- 恶性类固定为 `bcc/mel/scc`（ID `1/4/7`）；
- 良性组固定为其余五类（ID `0/2/3/5/6`）；
- 直接对现有 8 个 logits 分组做 `logsumexp`，再计算二组交叉熵；
- 仅训练阶段反向优化；验证阶段只报告同名 loss，不改变任何指标、fitness 或选择别名；
- 不增加参数或辅助头，验证与推理仍使用原 8 类 argmax。

权重 `0.5` 在训练前固定为主 8 类 CE 的半强度正则，不依据 Run29 验证结果调参。

## 保持不变

YOLO26m-sem 预训练资产、plain P3/P4 GAP+MLP 分类头、seed43、development
train/val、partial-label loader、采样、增广、batch=16、imgsz=640、AdamW、学习率、
100 epochs、epoch40 后 patience=30、验证指标、Joint、风险约束与 checkpoint v2 合同均与
Run22 相同。

## 预注册判定

主比较为 Run29 最佳 Joint bundle 对 Run22 最佳 Joint bundle。只有在不损害 IoU 和
Group Macro-F1 的前提下，MEL→NV 与 malignant→benign 风险有实质下降，并且 Joint 不低于
Run22，才建议继续做 seed42/44 配对复现。若整体效用下降或风险无改善，则停止该损失机制。
正式 feasible 仍必须满足冻结的 MEL→NV 上限 `0.1077844311`；未满足时不得锁为最终模型。

## 数据边界与入口

仅允许 development train/val。不得加载或评估 IMA++ test、ISIC2018 external 或
PAD-UFES-20。唯一正式入口为 `tools/run29/train_run29_from_pycharm.py`；默认安全
`resume=auto`。Codex 只执行预检，不调用 `trainer.train()`；正式启动必须由用户在 PyCharm 点击。
