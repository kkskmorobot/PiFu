# PiFuV2 当前交接（Run13 训练前）

日期：2026-08-08  
本文件是当前权威状态增量；此前所有 handoff、Run1~Run12 结果和 checkpoint 原样保留。

## 当前结论

- Run12 在可见 epoch 91 正常早停，91 行连续、非有限值 0；
- 5 个 Run12 bundle 完整，auto 指向内部 epoch 90，无异常/中断；
- Run12 最佳 binary IoU/Dice=0.77914/0.87586；最佳 image Macro-F1=0.55403；
  最佳 group Macro-F1/joint=0.59969/0.68258；
- Run12 最低 MEL->NV=0.25510，0 个 feasible checkpoint，未创建正式模型锁；
- 决策是不继续已正常早停的 Run12，改做 Run13 冻结 2×2 因素组合；
- Run13 已完成实现与全部训练前验收，正式训练尚未开始。

## Run13 研究因素

Run13 保持 Run12 BLG-MDA 注意力和所有数据/训练配置不变，只加入 Run11 的 MEL
target CE weight `5539/3047=1.8178536265178864`。它补全：

- Run10：无 attention / unweighted；
- Run11：无 attention / MEL weighted；
- Run12：attention / unweighted；
- Run13：attention / MEL weighted。

Run13 fresh start 于 `models/yolo26s-sem.pt`，不 warm-start Run12；因此相对 Run12
仍是严格单因素比较。

## 冻结训练与恢复

- train/val=23,462/1,744；manifest 与 Run12 逐字节一致；
- epochs=100、patience=20、batch=16、imgsz=640、workers=8、AdamW、AMP、seed=42；
- 完整 bundle 保存 model、EMA、optimizer、scaler、scheduler、stopper、selection、
  RNG、sampler/data progress；auto/explicit resume 拒绝不完整或指纹不一致 bundle；
- 25/25 tests、CPU resume、RTX 5080 AMP workers=2 resume、正式 Trainer 构造、
  真实 development batch forward/loss 全部 PASS；
- behavior fingerprint：`980198023b0119e63882eb5d5f5fedf662691eb4837a775c5b73b0696d38375b`。

## 门禁与下一动作

`PROJECT_RUN13_GATE=PASS`  
`RUN13_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

正式入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run13_from_pycharm.py`。

用户已于 2026-08-09 明确确认启动 Run13。下一动作只有一个：由用户在 PyCharm
运行该入口。不得修改预注册因素；正式模型锁前继续禁止 test；PAD-UFES-20 与
ISIC 2018 Task 1 仍仅为外部候选。
