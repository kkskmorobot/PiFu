# PiFuV2 当前交接（Run12 训练前）

日期：2026-08-06  
本文件是 2026-08-06 的状态增量；早期交接文档保留为历史证据，不覆盖、不删除。

## 当前权威状态

- Run1~Run9 原样保留；Run9 在 epoch 69 正常早停；
- Run10 正式训练与 val-only 统一评价已完成，作为 Run12 的受控基线；
- Run11 在 epoch 83（日志索引 82）正常早停，checkpoint integrity PASS，但没有满足 MEL->NV 上限的 feasible checkpoint，未建立正式模型锁；
- Run12 已完成预注册、模型实现、断点恢复验证与训练前 GPU 验收；正式训练尚未开始；
- test 从未用于 Run10/Run11/Run12 的模型开发或选择；
- PAD-UFES-20 和严格去重后的 ISIC 2018 Task 1 仍为外部候选，不参与训练。

## Run10 / Run11 对 Run12 的约束

Run10 的最佳 group Macro-F1 / joint 为 0.59426 / 0.68132，但最低 MEL->NV 为 0.32653，没有 feasible checkpoint。Run11 只把 MEL target CE 权重改为 1.8178536265，最低 MEL->NV 降到 0.11224，仍高于 0.1077844311；同时最佳 image Macro-F1、group Macro-F1、joint 分别下降 0.02513、0.02409、0.01878。

因此 Run12 回到 Run10 损失配置，只改变注意力聚合结构，避免把 Run11 的类别权重与新结构同时加入。研究因素明确对应“基于注意力机制改进 YOLO”：binary lesion-guided P3/P4/P5 dual attention（BLG-MDA）。

## Run12 冻结训练协议

- 数据：Run10 development manifest 的逐字节副本；train/val=23,462/1,744；
- 模型：YOLO26s semantic backbone + binary lesion-guided P3/P4/P5 dual-attention classifier；
- 训练：100 epochs，patience 20，batch 16，imgsz 640，workers 8，AdamW，AMP，device 0；
- 指标：沿用统一 val-only image/group 分类指标、binary segmentation 指标、joint score、MEL->NV 风险约束；
- resume：完整恢复 model、EMA、optimizer、scheduler、scaler、stopper、selection、RNG、sampler/batch order；拒绝不完整 bundle；
- 正式入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run12_from_pycharm.py`。

## 门禁与下一动作

`PROJECT_RUN12_GATE=PASS`  
`RUN12_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`

用户已于 2026-08-06 明确确认启动 Run12。下一动作仅有一个：从 PyCharm 运行 `train_run12_from_pycharm.py`。不得修改预注册因素；不得在正式模型选择前运行 test。
