# ab1 正式训练发布门

日期：2026-09-15（+08:00）

`PROJECT_AB1_GATE=PASS`

`AB1_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 发布结论

ab1 已按扩展计划阶段 D 完成预注册、代码隔离、数据边界、回归、CPU/CUDA 恢复、容量、
prepare-only 与正式 Trainer 构造检查。正式长训练未启动，正式结果目录不存在。用户可在 PyCharm
中运行唯一入口 `tools/ab1/train_ab1_from_pycharm.py`。

## 锁定身份

- seed=`43`；完整 LGLA 配对=`lgx`；Plain 共同参照=`Run22`；
- 唯一因素：把预测病灶概率图替换为全图常数 1；
- P3/P4、共享投影、双分支、softmax 门控、gate-balance `0.01`、分类器和参数量保持不变；
- config SHA-256=`7a56d935766258c1be5cbab20a4e8a8ae04d94dbcaeb6267b0ef63ab4ab3e213`；
- lock fingerprint=`5bc2f059c8d85d45aa6addd4496ad706ca7de03144cf9fe28d9aa0ad956ce122`；
- development manifest SHA-256=`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。

## 启动边界

本发布只授权用户亲自点击正式长训练。Codex 不得代替用户运行无参数正式入口。默认
`resume=auto`；首次点击时正式结果目录必须不存在。训练结束后先回到 Codex 审计 ab1，不自动准备
ab2、sum2 或 test2。
