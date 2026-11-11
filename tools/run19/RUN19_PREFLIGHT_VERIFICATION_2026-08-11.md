# Run19 预检复核

`RUN19_PREFLIGHT_VERIFICATION=PASS`

- 配置/单元测试：4 passed；确认继承冻结 Run16 注意力训练器、`joint_score-2` 门控、epoch 40 激活和激活后 stopper 状态 round-trip。
- CPU 中断恢复：PASS；431 个 live/EMA tensors、221 个 optimizer tensors、scheduler、stopper、selection、RNG 和下一 epoch 数据顺序最大差异均为 0。
- GPU AMP 中断恢复：PASS；RTX 5080，AMP checks passed，所有比较项最大差异 0。
- 本地 AMP 小模型：工作目录为 `tools`，使用 `tools/yolo26n.pt`，未发生下载。
- batch 容量：YOLO26m BLG-MDA，batch=16，imgsz=640，AMP forward/backward/AdamW step PASS；峰值 reserved 6,994,001,920 bytes，占设备显存 40.91%。
- 真实开发 batch 前向：PASS；P3/P4/P5、三个 identity-initialized ECA、detached lesion map、mask mix=0.5，模型状态未改变。
- 正式 Trainer 构造：PASS；`DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer`，未创建 optimizer，未调用 train。
- 回归测试：58 passed、2 subtests passed。
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`，与 Run18 相同。
- 配置 SHA-256：`03a0f9e2be6163cb9c51e06b3b4795e1f98a07e7e82755fd81eab61547051fb0`。
- ready 指纹：`e9bccce24376124422f226a4e7779faaaa8a9cea177c401e9a23b392fd76262c`。
- 正式训练：未启动；test/external metrics：未使用。
