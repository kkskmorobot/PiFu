# Run7 实现、验证与正式启动记录（2026-07-30）

## 项目决策

Run6 已于第 57 轮正常早停，最佳 checkpoint 为第 37 轮。其统一 val
Macro-F1 为 0.701557，但 foreground mIoU 仅 0.489603，未达到预注册下限
0.518748。按照交接文档的 runs5~runs8 消融矩阵，下一步执行 Run7，而不是
继续 Run6 或提前修改 loss/sampler。

## Run7 唯一主要变化

从 Run5 底座出发，仅替换图像分类特征聚合：

- 语义主头生成 `1 - P(background)` 软前景概率；
- 前景概率分别引导 P3/8、P4/16、P5/32 的归一化加权池化；
- 三个尺度各使用一个 identity-initialized ECA 风格通道门；
- 每尺度描述子为 50% 全局平均池化与 50% 病灶引导池化；
- 前景 mask 在分类分支停止梯度，不增加一致性 loss；
- 语义 P3/P4 主/辅助路径保持 Run5 原样；
- 数据、预训练权重、loss、采样、增强、优化器、seed 与训练参数均不变。

loss 与 lesion-balanced sampler 仍保留给 Run8，避免与结构改进混杂。

## 已通过验证

- Python 语法编译通过。
- 数据 dry-run 通过：train 8512、val 1504、test 1503；只做结构计数，未做 test 推理。
- 三尺度接线：head sources 为 `[16, 13, 10]`，模型缓存包含 P5 layer 10。
- 通道数：P3/P4/P5 为 128/256/512；恰有三个 ECA 模块。
- 参数量：Run5 6,550,487；Run6 6,552,633；Run7 6,616,038；Run7 相比 Run5 增加 65,551。
- P5 分类器输入块初始为零贡献，原 P3/P4 分类器初始化被保留。
- 严格确定性 CUDA 前向/反向通过；训练输出形状为主语义 `(2,8,16,16)`、辅助语义 `(2,8,8,8)`、分类 `(2,7)`。
- 1% 数据、1 epoch GPU smoke 通过：85 张 train，完整 1504 张 val；四项 loss、最终验证、权重 strip/reload 与 checkpoint selection 均正常。
- smoke 生成的 85 图 `train.cache` 已删除；正式启动后重建的完整缓存为 7,386,552 bytes，SHA-256 与 Run6 前记录一致：`AB7B14E564EDFD5CDFCC86A02082B8017D121F70766E9BE80BC9DA1AABC4153D`。
- 重复下载到工作区的预训练权重已在确认与项目正式权重 SHA-256 完全一致后删除；Run7 入口现将模型路径固定解析到 PiFuV2 根目录。

## 正式训练启动证据

- PyCharm 入口：`tools/train_skin7_yolo26_lesion_guided.py`
- 配置：`experiments/07_YOLO26s_Semantic_LesionGuidedDualAttention/train_yolo26s_sem_cls_lesion_guided_v7.yaml`
- 输出：`experiments/07_YOLO26s_Semantic_LesionGuidedDualAttention/results/semantic_multitask_lesion_guided/skin7_yolo26s_sem_lesion_guided_seed42`
- `args.yaml` 已落盘：model 为项目内绝对路径；epochs=100、patience=20、batch=16、imgsz=640、device=0、seed=42、deterministic=true、split=val。
- 正式 train/val 分别扫描 8512/1504 张且无损坏样本。
- 观察到 epoch 1/100 已运行至 52/532（约 9%），GPU 利用率 100%，总显存占用约 7.96 GB。
- 正式训练继续在 PyCharm 中运行；本记录未等待其完成。

## 中断恢复与选择协议

仅在真正中断且 `last.pt` 含完整 optimizer 状态时，从 PyCharm 运行
`tools/resume_yolo26_lesion_guided_v7.py`。Ultralytics 8.4.101 恢复 optimizer、
scheduler、scaler、EMA 和 global best fitness，但 EarlyStopping 历史耐心计数
不会完整恢复。

继续保存同协议 `best.pt`、`best_macro_f1.pt` 与满足全部约束时才产生的
`best_feasible_macro_f1.pt`。最终方法冻结前不使用 test 指标。
