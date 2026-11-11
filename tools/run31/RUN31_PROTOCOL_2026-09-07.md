# Run31 预注册协议：plain seed43 + MEL/NV 训练期辅助头

`RUN31_PROTOCOL_STATUS=PREREGISTERED`

## 目的与唯一因素

Run30 把 MEL→NV 从 Run22 的 `0.40816` 降至 `0.30612`，但最佳 Joint 同时下降
`0.01412`、Group Macro-F1 下降 `0.03113`，没有通过“风险实质改善且整体效用无明显损害”的
预注册门，因此停止该 posterior focal risk loss，不根据验证结果调 gamma 或权重。

Run31 回到未修改的 Run22 plain seed43，唯一新增因素是一个训练期 MEL/NV 辅助任务：

- 从 YOLO26m 主图像分类器已经存在的 256 维隐藏描述符接出 `Linear(256, 2)`，仅增加 514 个参数；
- 辅助类固定为 NV=`0`、MEL=`1`；仅用官方标签为 MEL 或 NV 的训练行；
- 辅助交叉熵按 MEL/NV 行数归一化，并以 `0.5` 加入原图像分类目标；
- 主 8 类 CE、主 logits、验证指标、checkpoint 选择与推理 argmax 均不变；
- eval/inference 不输出也不使用辅助 logits。

该机制通过共享特征提供层级监督，而不直接惩罚或重排主 8 类 logits；与 Run11 全 MEL CE 加权、
Run14 margin、Run15 main-logit pairwise loss、Run29 二组 NLL、Run30 posterior focal loss 均不同。

## 固定项

YOLO26m-sem、plain P3/P4 GAP+MLP、seed43、development train/val、partial-label loader、
采样、增广、batch16、imgsz640、AdamW、学习率、100 epochs、epoch40 后 patience30、
checkpoint v2、风险上限与 utility floor 全部保持 Run22 不变。

## 预注册判定

Run31 最佳 Joint bundle 与 Run22 最佳 Joint bundle做严格同 seed43 配对。只有 MEL→NV 出现实质
改善，且 Joint 与 Group Macro-F1 不出现明显损害，才考虑 seed42/44 复现。若风险改善不足或效用
明显下降，则停止该辅助头，不在训练后依据 validation 调辅助权重、隐藏层或类别定义。正式 feasible
仍要求 MEL→NV≤`0.1077844311`。

## 边界与入口

仅允许 development train/val；不得加载 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
唯一正式入口为 `tools/run31/train_run31_from_pycharm.py`，默认 `resume=auto`。Codex 仅做发布
预检，不调用正式 `trainer.train()`；启动必须由用户在 PyCharm 点击。
