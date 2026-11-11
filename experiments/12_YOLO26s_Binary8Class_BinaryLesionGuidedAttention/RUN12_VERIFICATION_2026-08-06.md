# Run12 实现与训练前验收

日期：2026-08-06  
实验：`experiments/12_YOLO26s_Binary8Class_BinaryLesionGuidedAttention`

`RUN12_IMPLEMENTATION_GATE=PASS`  
`RUN12_STATIC_TEST_GATE=PASS`  
`RUN12_CPU_RESUME_GATE=PASS`  
`RUN12_GPU_RESUME_GATE=PASS`  
`RUN12_FORMAL_CONSTRUCTOR_GATE=PASS`  
`RUN12_REAL_BATCH_FORWARD_GATE=PASS`  
`RUN12_TEST_DATA_GATE=NOT_TOUCHED`  
`RUN12_EXTERNAL_DATA_GATE=NOT_TOUCHED`  
`RUN12_FORMAL_TRAINING_STARTED=FALSE`

## 1. 受控实验结论

Run12 以 Run10 为基线，只改变分类特征聚合机制：将原来的 P3/P4 普通全局池化分类头，替换为二值病灶引导的 P3/P4/P5 多尺度双注意力分类头（BLG-MDA）。

保持不变：

- 数据 manifest、train/val split、mask 值域和 test 规则；
- semantic main/aux 分支、分割损失和 partial-label 规则；
- 分类损失权重、优化器、学习率、增强、batch、imgsz、epochs、patience 和统一 val 协议；
- Run10 的 P3/P4 分类头初始化；
- Run11 的 MEL 加权 CE 不继承，因为它是另一项已完成的诊断因素。

新增机制：

- P3/P4/P5 三尺度 identity-initialized ECA 通道注意力；
- 从 binary semantic main logit 得到 detached soft lesion map；
- 每个尺度使用 50% global average 与 50% normalized lesion-guided average；
- P5 新增输入块零初始化，避免在训练开始前改变 Run10 分类输出；
- 三尺度描述符连接后进入与 Run10 同宽度、同 SiLU、同 dropout=0.2 的 8-logit MLP。

## 2. 初始化等价性与规模

- 完整模型参数：6,614,361；
- 分类/注意力头参数：115,848；
- 相比 Run10 总参数增加：65,551；
- attention source indices：`[16, 13, 10]`；
- scale channels：`[128, 256, 512]`；
- pretrained transfer：360/371 items；
- 同一随机种子下，semantic main/aux 权重与 Run10 逐元素一致；
- Run10 P3/P4 分类初始化逐元素一致；
- P5 输入块为零，3 个 ECA 模块在初始化时均为精确恒等映射。

## 3. 自动化验证

- `py_compile`：PASS；
- `unittest discover`：19/19 PASS，其中 Run12 新增 7 项，并覆盖 Run10/Run11 回归与 checkpoint contract；
- CPU 完整模型 forward/backward：PASS；
- CPU 连续训练与中断恢复等价：model、EMA、optimizer、scheduler、stopper、selection、RNG、下一 epoch batch order 的最大差值均为 0；
- RTX 5080 + AMP + workers=2 中断恢复等价：上述状态最大差值均为 0；
- 正式 Trainer 构造检查：epochs=100、patience=20、batch=16、imgsz=640、workers=8、AdamW、AMP、device=0；未创建 optimizer，未调用 train；
- 真实 development role batch forward/loss：batch=16、imgsz=640、AMP PASS；未 backward、未 optimizer step、未调用 trainer.train；
- GPU：NVIDIA GeForce RTX 5080，Torch 2.8.0+cu129，CUDA 12.9；
- dry-run 峰值：allocated 600,452,608 bytes，reserved 1,004,535,808 bytes。

## 4. 数据边界核验

- development train：23,462（seg 10,483；cls 13,715；both 736）；
- development val：1,744（seg 292；cls 1,488；both 36）；
- IMA++ source test 634 行全部排除；
- development manifest 不含 test key；
- external candidates 不在 development manifest；
- 手工解压 PAD-UFES-20 三部分共 2,298 张 PNG，CRC32/size PASS，但训练角色仍为 `none_external_candidate_only`。

## 5. 冻结标识

- config SHA-256：`90e1ac7fe0a04afe571ca4c6714ad5dca12d5b8e770849944528dff3a8c33b2f`；
- protocol SHA-256：`a3d0b05c6abc4c811ffac660887b27afc99b963dbd6259f7540ecb0e3538ac76`；
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`；
- entrypoint SHA-256：`0321850a48d3ea59a72c4dd5982df16d10396be0bb940242a3a16af10cbbed7a`；
- model module SHA-256：`dc32caaff2e840f1e928bec011c9b256c5319da33ccc95afaaa60f1ec6222c27`；
- behavior fingerprint：`77ba2344a10784443d11ded2a9e6dff497e92429481a986a3539becdbc45710f`。

以上只证明 Run12 已达到“可由用户在 PyCharm 点击正式训练”的工程状态，不代表正式训练已经开始，更不代表模型选择或 test 评价已经获准。
