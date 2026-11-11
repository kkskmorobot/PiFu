# Run23 训练前预检复核

`RUN23_PREFLIGHT_VERIFICATION=PASS`

`RUN23_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN23_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 已通过

- Run22 正常结束复核：CSV epoch 1~93；训练完成记录存在；没有遗留 Python 训练进程。
- Run22 五个保留 checkpoint bundle 的 `_SUCCESS`、大小、SHA-256 和 fingerprint 全部通过。
- Run23 配置 SHA-256：`6e85f7544c034960ffaf68e8a9268b15367c311cb484c301ed52f19f698ebf44`。
- Run23 当前就绪 fingerprint：`8837f6958bf63820732ad852367dd2d638927fcdd9de0d0a96c039c2cdf55c13`。
- Run23 模块测试与 Run18 延迟早停回归测试：7 passed。
- CPU 合成数据完整状态中断/恢复：PASS；live model、EMA、optimizer、RNG、下一 epoch batch order 等比较的最大绝对差均为 0。
- GPU/AMP 合成数据完整状态中断/恢复：PASS；RTX 5080 上连续训练与中断恢复后的 live model、EMA、optimizer、RNG 和下一 epoch batch order 最大绝对差均为 0。
- YOLO26m 正式容量预检：batch=16、imgsz=640、AMP forward/backward/AdamW step PASS；峰值 reserved 显存 6,996,099,072 bytes（设备显存的 40.93%），optimizer step 已完成。
- 模型结构：14,507,348 参数；分类头 198,920 参数；相对 plain 头仅增加 2 个 kernel size 5 ECA 模块，共 10 个注意力参数；不含空间注意力和 P5 描述符。
- development manifest 与确定性投影一致：train 23,462、val 1,744；IMA++ test 634 行排除；YAML `test: null`；外部候选不存在。
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。
- development YAML SHA-256：`7aeb708750650742b46f06fc976d9db7db1bad68676c7ce2fde2f6f65beb5e62`。
- 8 个受保护输入哈希通过；CUDA 可见 1 张 `NVIDIA GeForce RTX 5080`，PyTorch `2.8.0+cu129`，CUDA runtime `12.9`。
- 正式入口 `--prepare-only`：PASS；`RUN23_PREFLIGHT_REPORT.json` 与 `RUN23_READY_TO_TRAIN.json` 已生成，`trainer.train()` 未调用。
- 正式 Trainer 只构造：PASS；模型类 `P3P4ECAMultiTaskModel`，训练器 `DelayedUtilityGatedP3P4ECATrainer`，优化器未创建，`trainer.train()` 未调用。
- Run23 正式实验目录不存在；未运行正式训练、test 或外部评价。

## AMP 本地资产修复记录

首次 GPU/AMP 恢复预检从项目根目录运行时，Ultralytics 自检误下载了一个根目录 `yolo26n.pt`。该文件与锁定的 `tools/yolo26n.pt` 大小均为 5,544,453 bytes，SHA-256 均为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

已给 `preflight_run23_resume_v1.py` 增加 `os.chdir(TOOLS_DIR)`，随后重新完成 GPU/AMP 恢复预检并 PASS；正式入口本身也在 main 前切换到 `tools`，最终 lock inputs 明确使用 `tools/yolo26n.pt`。根目录的字节相同副本未擅自删除，也不属于 Run23 fingerprint。

## 当前发布状态

训练前技术门已全部通过。用户已于 2026-08-15 明确回复“确认启动 Run23 正式训练”，发布门已改为 PASS。可以从 PyCharm 运行 `tools/run23/train_run23_from_pycharm.py`；本次仅解锁发布门，尚未从命令行启动正式训练。
