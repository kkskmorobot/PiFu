# Run25 训练前预检复核

`RUN25_PREFLIGHT_STATUS=PASS`

`RUN25_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN25_GATE=BLOCKED_PENDING_USER_CONFIRMATION`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 冻结标识

- 配置 SHA-256：`05cec4a8dde18e93858d18b352b8429edc4ed592f14db65c56f680ca5969927b`。
- 最终 fingerprint：`c2cf59e05a941c81d744bc47e722cebcc61bd24da715cfe48e34970048dd6910`。
- Run25 正式实验目录不存在。

## 已通过项目

- Run25 模块测试与 Run18 延迟早停回归：7 passed。
- development manifest 与既有冻结投影逐字节一致；train=23,462、val=1,744；test 为 null，PAD-UFES-20 与外部候选不在开发入口。
- CPU 合成数据完整状态恢复：PASS；连续与中断恢复后的 live model、EMA、optimizer、scheduler、stopper、selection、RNG 和第二轮 batch 顺序均 0 差异。
- RTX 5080、CUDA 12.9、AMP、workers=2 合成数据完整状态恢复：PASS；上述比较均 0 差异。
- YOLO26m batch=16、imgsz=640、AMP 单步容量：PASS；14,507,348 参数，分类头 198,920 参数，两个 ECA 共 10 参数；峰值预留显存 6,996,099,072 bytes，占设备显存 0.40994；完成 AdamW step。
- `--prepare-only`：PASS；真实 development batch 前向与损失有限；未创建 optimizer、未 backward、未训练。
- 正式 Trainer 只构造：PASS；seed=44、epochs=100、batch=16、workers=8、AdamW、AMP；模型类 `P3P4ECAMultiTaskModel`，分类头含两个 identity-initialized ECA；优化器未创建，`trainer.train()` 未调用。

## 结论

Run25 技术门已全部通过，入口已经准备到只差用户确认。发布文档必须在用户明确确认后记录 `USER_CONFIRMATION_RECORDED=TRUE` 并将项目门改为 PASS，之后才允许在 PyCharm 中运行正式入口。仍不得运行 test 或外部候选评价。
