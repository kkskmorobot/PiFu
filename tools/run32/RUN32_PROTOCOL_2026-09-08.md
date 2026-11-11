# Run32 预注册协议：plain seed43 + 恶性/良性训练期辅助头

`RUN32_PROTOCOL_STATUS=PREREGISTERED`

## 目的与唯一因素

Run31 的 MEL/NV 窄辅助头对 MEL→NV 仅改善 `0.04081`，同时使 Group Macro-F1 下降
`0.07008`、Joint 下降 `0.03456`，因此停止该机制。Run32 回到未修改的 Run22 plain seed43；
唯一新增因素为恶性/良性层级辅助任务：

- 从主图像分类器已有 256 维隐藏描述符接出 `Linear(256,2)`，新增 514 个参数；
- benign=`akiec/bkl/df/nv/vasc`，malignant=`bcc/mel/scc`，严格来自冻结 taxonomy；
- 对全部官方分类训练行计算二类 CE，权重因覆盖全部标签而预先固定为 `0.25`；
- 主 8 类 CE、主 logits、验证指标、checkpoint 选择与推理 argmax 均不变；
- eval/inference 不输出也不使用辅助 logits。

该机制不同于 Run29 在主 8 类概率上直接增加 group NLL，也不同于 Run31 只覆盖 MEL/NV 的窄
辅助任务。它用完整标签密度提供层级特征监督，但不直接重排主 8 类 logits。

## 固定项与判定

YOLO26m-sem、plain P3/P4 GAP+MLP、seed43、development train/val、partial-label loader、采样、
增广、batch16、imgsz640、AdamW、学习率、100 epochs、epoch40 后 patience30、checkpoint v2、
风险上限与 utility floor 全部保持 Run22 不变。

Run32 最佳 Joint bundle 与 Run22 最佳 Joint bundle做严格配对。只有 MEL→NV 和
malignant→benign 出现实质改善，且 Joint 与 Group Macro-F1 不出现明显损害，才考虑 seed42/44
复现。若风险改善不足或效用明显下降，则停止训练期风险辅助路线，进入 development-only 固定代价
决策研究；不得在训练后依据 validation 调辅助权重、类别定义或隐藏层。

## 边界与入口

仅允许 development train/val；不得加载 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
唯一正式入口为 `tools/run32/train_run32_from_pycharm.py`，默认 `resume=auto`。Codex 只做发布
预检，不调用正式 `trainer.train()`；启动必须由用户在 PyCharm 点击。
