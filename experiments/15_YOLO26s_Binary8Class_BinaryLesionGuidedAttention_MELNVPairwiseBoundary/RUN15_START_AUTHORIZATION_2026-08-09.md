# PiFuV2 Run15 正式训练启动授权

日期：2026-08-09  
用户确认原文：`确认启动 Run15 正式训练`

`RUN15_FORMAL_TRAINING_AUTHORIZED=TRUE`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_EVALUATION_AUTHORIZED=FALSE`  
`EXTERNAL_EVALUATION_AUTHORIZED=FALSE`

授权范围仅为冻结 Run15 的 train/val 正式训练：

- 入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run15_from_pycharm.py`；
- behavior fingerprint：
  `f2e38bd9fd0422280de8c1e8ac3520d5bf6f0843efaef343d7e0601f3a43b978`；
- 从 PyCharm 无参数启动，默认 `resume=auto`；
- 当前无完整 Run15 bundle 和正式结果目录，首次启动必须从冻结 base weights fresh start；
- 以后只可从通过大小、SHA-256、fingerprint、完整状态和 `_SUCCESS` 校验的 Run15
  bundle 恢复；
- 本授权不包含 test 或 external evaluation，正式模型锁前继续禁止二者。
