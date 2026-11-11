# Run30 预注册协议：plain seed43 + true-MEL→NV posterior focal risk

`RUN30_PROTOCOL_STATUS=PREREGISTERED`

## 目的与唯一因素

Run29 的对称二组 NLL 同时约束大量良性样本，Group Macro-F1 和 Joint 明显下降，却仅轻微改善
MEL→NV。Run30 回到未修改的 Run22 plain seed43；唯一训练因素是在原 8 类无权重 CE 上增加：

`0.5 × mean_true_MEL[p_NV^2 × -log(1-p_NV)]`

- `p_NV` 来自现有 8 类 softmax；gamma 固定为 `2.0`；
- 只对官方标签为 MEL 的行计算，并按 true-MEL 行数归一化；
- 非 MEL 行的辅助梯度严格为 0；
- 高 NV 概率的危险 MEL 样本获得更强惩罚，低风险 MEL 样本影响趋近于 0；
- 不增加参数，不改变 8 类 CE、验证指标、选择规则或推理 argmax。

该机制不同于 Run11 的全 MEL CE 权重、Run14 的八类 logit margin、Run15 的 MEL/NV pairwise
margin，也不同于 Run29 的全样本对称恶性/良性 NLL。

## 固定项

YOLO26m-sem、plain P3/P4 GAP+MLP、seed43、development train/val、partial-label loader、
采样、增广、batch16、imgsz640、AdamW、学习率、100 epochs、epoch40 后 patience30、
checkpoint v2、风险上限与 utility floor 全部保持 Run22 不变。

## 预注册判定

Run30 最佳 Joint bundle 与 Run22 最佳 Joint bundle做严格配对。只有 MEL→NV 出现实质改善，且
Joint 与 Group Macro-F1 不出现明显损害，才准备 seed42/44 复现。若风险改善不足或效用再次明显
下降，则停止该损失，不在训练后调 gamma/weight。正式 feasible 仍要求 MEL→NV≤`0.1077844311`。

## 边界与入口

仅允许 development train/val；不得加载 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
唯一正式入口为 `tools/run30/train_run30_from_pycharm.py`，默认 `resume=auto`。Codex 仅做发布
预检，不调用正式 `trainer.train()`；启动必须由用户在 PyCharm 点击。

