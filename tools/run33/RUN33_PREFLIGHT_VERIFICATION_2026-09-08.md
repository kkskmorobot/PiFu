# Run33 发布前预检记录（2026-09-08）

`RUN33_PREFLIGHT_STATUS=PASS`

## 已通过项目

1. 目标目录 Python 静态编译：PASS。
2. 标准 CBAM、梯度、参数校验和延迟早停单元测试：6/6 PASS。
3. `--prepare-only`：PASS；冻结 development manifest/YAML、protected SHA-256、taxonomy、
   test/external 边界全部通过。
4. 真实 development role batch16、imgsz640、AMP 前向与 partial-label loss：PASS；模型状态未改变，
   未创建优化器、未反向、未调用 `trainer.train()`。
5. 正式 Trainer 构造：PASS；模型为 `P3P4StandardCBAMMultiTaskModel`，分类头为
   `P3P4StandardCBAMClassificationHead`，P3/P4 各一个标准 CBAM，语义分支不修改。
6. 正式参数锁：seed43、YOLO26m-sem、batch16、imgsz640、workers8、AdamW、AMP、100 epochs、
   epoch40 后 patience30、unweighted CE、image weight0.5、统一指标/选择合同均通过。
7. RTX 5080 合成容量：PASS；完成 batch16/img640/AMP 前向、反向和 AdamW step；峰值 reserved
   `7,077,888,000` bytes，占设备显存约 `41.47%`。
8. 模型参数：总参数 `14,548,494`；原图像分类 MLP `198,920`；两个标准 CBAM 共新增
   `41,156`，其中通道注意力 `40,960`、空间注意力 `196`。
9. 合成中断/恢复：PASS；live model、EMA、optimizer、RNG 与第二 epoch batch 顺序零差异，
   checkpoint bundle 的 `_SUCCESS`、fingerprint 和状态恢复均通过。
10. 正式结果目录不存在；正式训练未启动；IMA++ test、ISIC2018 external、PAD-UFES-20 均未使用。

## 锁定值

- 配置 SHA-256：`9dc2ab8135ed32709998d0ed086400d098719d982be63a2ee623b527b57c9ae7`
- Run33 fingerprint：`95d6e3a2fb3bb5aa05ce36ababead5f9f872272a1890e7bca03bdc96c94c533e`
- 正式入口：`tools/run33/train_run33_from_pycharm.py`
- 正式结果目录：
  `experiments/33_YOLO26m_Binary8Class_P3P4_StandardCBAM_Seed43/results/run33_yolo26m_p3p4_standard_cbam_seed43`

