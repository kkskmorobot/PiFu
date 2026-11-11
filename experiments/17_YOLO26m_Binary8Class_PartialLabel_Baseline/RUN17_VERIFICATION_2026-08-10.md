# Run17 正式训练前验证

`RUN17_VERIFICATION_STATUS=PASS`

## 合同与数据

- 正式角色：YOLO26m 无注意力 partial-label 多任务基线；相对 Run16 唯一因素为删除 BLG-MDA；
- 配置 SHA-256：`f9d684f31194ad98cc2e3560b0299327bc409f4a6543a109d06a8b4f63c6f2c9`；
- ready 行为指纹：`cfd3404510f226a9059ec3cb9408ed95df5580a18da4ced74a416555aee5d4b8`；
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`，与 Run16 字节一致；
- train：23462（seg 10483，cls 13715，both 736）；val：1744（seg 292，cls 1488，both 36）；
- source test 634 条排除，development YAML 的 `test` 为 null，外部候选不存在于训练配置；
- PAD-UFES-20 手工解压 2298 个 PNG 的大小和 CRC32 与三个 ZIP 一致，训练角色仍为 none。

## 模型与容量

- RTX 5080、Torch 2.8.0+cu129、CUDA 12.9；
- 真实 development role 样本扩展到 batch=16、imgsz=640、AMP 的前向与损失均有限；
- 正式模型：`BinaryPartialMultiTaskModel`，分类头：`BinaryPartialClassificationHead`；
- 参数 14,507,338，分类头参数 198,920；描述符仅 P3/P4 global average pooling；确认无 attention 属性；
- yolo26m-sem 预训练权重转移 420/428；
- 一次完整 synthetic batch=16、640、AMP 前向、反向和 AdamW step 通过；峰值 allocated 6,455,815,168 bytes，reserved 6,996,099,072 bytes，占 16 GB 设备约 40.93%；
- 容量检查未加载 train/val/test 图像，也未启动正式训练。

## 测试与恢复

- py_compile：PASS；
- 基础多任务 + Run17 选择器测试：10 passed，2 subtests passed；
- CPU 连续两轮与安全中断/恢复：PASS，live/EMA/optimizer/RNG/下一轮样本顺序最大差异均为 0；
- RTX 5080 + AMP + workers=2 连续/恢复：PASS，live 428、EMA 428、optimizer 218 个 tensor 的最大差异均为 0；
- 正式 Trainer v2 构造：PASS；epochs=100、patience=20、batch=16、imgsz=640、workers=8、AdamW、AMP、device=0；未创建 optimizer，未调用 `trainer.train()`；
- 构造器 v1 因预检脚本在 trainer setup 前把模型路径字符串当作模型对象而失败；已保留失败记录，v2 显式调用正式 `get_model()` 路径后通过，正式入口未受影响。

## 旧 Run17 边界草案

旧边界注意力草案从未建立 results 目录、从未正式训练。其 11 个文件已复制归档至 `preflight/superseded_run17_boundary_draft_20260810_v1`；旧 ready marker 作废。正式 Run17 唯一有效实验目录为 `experiments/17_YOLO26m_Binary8Class_PartialLabel_Baseline`。

`FORMAL_TRAINING_STARTED=FALSE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`
