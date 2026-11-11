# Run18 预检复核

`RUN18_PREFLIGHT_VERIFICATION=PASS`

- 早停单元测试：3 passed；epoch 1–39 不计 patience，epoch 40 激活，连续无改进时最早 epoch 70 停止。
- CPU 断点恢复：PASS；live model、EMA、optimizer、scaler、scheduler、stopper、selection、RNG、下一 epoch 数据顺序均无差异。
- GPU AMP 断点恢复：PASS；RTX 5080，所有比较项最大差异 0，AMP checks passed。
- AMP 小模型：工作目录为 `tools`，使用 `tools/yolo26n.pt`；受保护哈希为 `9b09cc8b...ad4fef`，未发生下载。
- batch 容量：YOLO26m，batch=16，imgsz=640，AMP 前向/反向/AdamW step PASS；峰值 reserved 6,996,099,072 bytes，占设备显存 40.93%。
- 正式 prepare-only：PASS；development manifest=23,462 train / 1,744 val，未包含 test 或外部候选。
- 正式 Trainer 构造：PASS；`DelayedUtilityGatedPartialLabelMultiTaskTrainer` + `BinaryPartialClassificationHead`，未创建 optimizer，未调用 train。
- 回归测试：正式 `tests` + Run18 测试为 54 passed、2 subtests passed。无范围限定的 pytest 会收集历史归档 Run17 草案中的同名测试并产生 import mismatch；历史归档按保留规则未删除。
- 配置 SHA-256：`e0f6a2154d7bd8cfe85a7eb6f2ebcc4e5458d597832780450bd6cde29108265b`。
- ready 指纹：`2ed64c4aa90c6d90d15ace2158de0866e570f71ad8a3cb3bbc3e2b7d0fd2b7a3`。
- 正式训练：未启动。
- test/external metrics：未使用。
