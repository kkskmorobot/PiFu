# Run26 预检复核记录

复核日期：2026-09-02

`RUN26_PREFLIGHT_VERIFICATION=PASS`

`RUN26_READY_TO_CLICK_GATE=PASS`

`PROJECT_RUN26_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 锁定标识

- 配置 SHA-256：`60f24b15ec51995f9728e86777bda3dcf6274bb7feac5cfd92652ed7e4500546`
- 最终就绪指纹：`a440204a90d71e79e1d1c9cfa25fadcdc7a0c76854db469a01387bae6451abd7`
- Run25 基线指纹：`c2cf59e05a941c81d744bc47e722cebcc61bd24da715cfe48e34970048dd6910`

## 复核结果

1. Python 语法编译通过。
2. Run26 静态测试及 Run18 延迟早停回归测试共 `8 passed`。
3. CPU 合成两轮断点恢复复核通过：模型、EMA、优化器、调度器、早停器、选择状态、随机数状态和 batch 顺序与连续训练逐项零差异。
4. RTX 5080 GPU/AMP/workers=2 断点恢复复核通过，上述状态同样逐项零差异。
5. 容量预检通过：batch=16、imgsz=640、AMP、AdamW 更新成功；模型参数 14,507,544，分类头参数 198,920，新增注意力参数 206，峰值 reserved 显存 7,077,888,000 字节，占显存约 41.47%。
6. `--prepare-only` 使用开发清单真实 batch 完成，并写出就绪标记；未启动正式训练。
7. 正式 Trainer 构造通过：seed=44、epochs=100、patience=30、batch=16、imgsz=640、workers=8、AdamW、AMP、device=0；未创建优化器，未调用 `trainer.train()`。
8. 模型结构复核通过：P3/P4 各一个 ECA 和一个 7×7 恒等初始化空间门，无 P5 注意力，语义分支不变。
9. 正式 Run26 实验目录不存在，未产生任何正式训练结果。
10. 未运行 test，未使用外部候选指标，未下载任何模型或数据。

## 发布结论

用户已于 2026-09-02 明确回复“确认启动 Run26 正式训练”。确认已记录，发布门已切换为 PASS，允许从锁定的 PyCharm 入口脚本启动一次正式训练。
