# PiFuV2 Run14 首轮断电恢复记录

日期：2026-08-09  
事件：首次正式训练在第 1 个 epoch 完成前因电脑意外断电终止。

`RUN14_POWER_LOSS_RECOVERY_GATE=PASS`  
`COMPLETE_RESUME_BUNDLE_FOUND=FALSE`  
`CURRENT_RESUME_MODE=FRESH_START_NO_COMPLETE_BUNDLE`  
`TRAINING_STARTED_BY_RECOVERY=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 只读诊断

- `run14_development_manifest.csv` 存在但为 0 字节；预期 8,299,971 字节；
- `RUN14_PROTOCOL_2026-08-09.md` 同样被截成 0 字节；
- 其余冻结 lock inputs 全部与 `RUN14_READY_TO_TRAIN.json` 中训练前 SHA-256 匹配；
- 首轮目录仅含 `args.yaml`、launch record、空 `weights` 目录和 4 个 0 字节预览图；
- 不存在 `checkpoint_bundles`、`_SUCCESS`、model/EMA/optimizer/scaler/scheduler/RNG
  等完整训练状态，因此禁止把它当作可恢复断点。

## 2. 可恢复归档与恢复

断电现场已原样保留在：

`experiments/14_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELNVLogitMargin/crash_archives/power_loss_pre_epoch1_20260809_155500`

归档包含两个 0 字节冻结文件副本和完整半成品 formal result directory。随后从训练前
已验证暂存副本恢复：

- protocol：3,538 字节，SHA-256=
  `22ee6959ff1f795b230b91a9bff4a8c38b15b4220e30c837336dae9287976bc6`；
- development manifest：8,299,971 字节，SHA-256=
  `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。

没有改动 source data manifest、split、mask value domain、test 规则或 external candidate。

## 3. 恢复后复核

- release、locked config、development artifacts 和全部 lock inputs 验证 PASS；
- config SHA-256=`7d50195c4713656a3d1e02e0f450018dce8339cae7019a09ecdc9ebf56d31a0a`；
- behavior fingerprint=
  `30e88fffb1eab5c59d7a3a1e0cbb458dcd2b020414c1011853941ce9fd456bc4`；
- RTX 5080 / CUDA 可用；当前 formal result directory 与 bundle root 均不存在；
- 恢复过程只调用门禁和哈希验证，没有构造 Trainer、没有启动训练、没有使用 test/external。

## 4. 下一动作

从 PyCharm 无参数重新运行 `train_run14_from_pycharm.py`。默认 `resume=auto` 会因没有
完整 bundle 而安全地 fresh start。只有至少完成一个 epoch 并原子发布 `_SUCCESS`
bundle 后，未来再次运行才可进行完整状态恢复。
