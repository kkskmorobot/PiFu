# Run16 实现与训练前验收

日期：2026-08-09  
实验：`experiments/16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention`

`RUN16_IMPLEMENTATION_GATE=PASS`  
`RUN16_STATIC_TEST_GATE=PASS`  
`RUN16_CPU_RESUME_GATE=PASS`  
`RUN16_GPU_RESUME_GATE=PASS`  
`RUN16_FORMAL_CONSTRUCTOR_GATE=PASS`  
`RUN16_REAL_BATCH_FORWARD_GATE=PASS`  
`RUN16_CAPACITY_GATE=PASS`  
`RUN16_TEST_DATA_GATE=NOT_TOUCHED`  
`RUN16_EXTERNAL_EVALUATION_GATE=NOT_TOUCHED`  
`RUN16_FORMAL_TRAINING_STARTED=FALSE`

## 1. 受控实验合同

相对 Run12，唯一训练因素是将 `models/yolo26s-sem.pt` 替换为官方
`models/yolo26m-sem.pt`。Run12 的二值语义头、BLG-MDA、unweighted CE、loss weight、
数据、采样、增强、batch、imgsz、optimizer、LR、epochs、patience 与 seed 均不变。
Run13/14/15 的 MEL weighted CE、logit margin 和 pairwise loss 均未继承。

Run15 的 utility floors 仅作为 val checkpoint eligibility 纠错保留：binary IoU>=0.70、
group Macro-F1>=0.50；原 fitness 未改变，因此不构成第二个训练因素。

## 2. 权重、规模与真实前向

- 官方权重大小：28,924,971 bytes；
- 权重 SHA-256：
  `3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233`；
- actual pretrained transfer：420/431 items；
- Run16 完整模型参数：14,638,425；Ultralytics summary：59.7 GFLOPs；
- classification head 参数：329,992；P3/P4/P5 channels=[256,512,512]；
- 真实 development role batch=16、imgsz=640、AMP forward/loss PASS；
- semantic/classification shape=[16,1,80,80]/[16,8]；loss 四项均有限；
- 前向峰值 allocated/reserved=1,064,696,832/1,864,368,128 bytes；
- 模型 state 未改变；未 backward、未创建 optimizer、未 optimizer step、未调用 train。

另以随机初始化同结构执行合成 forward/backward/AdamW 单步容量检查：batch=16、640、
AMP 的峰值 allocated/reserved=6,463,043,584/6,994,001,920 bytes，RTX 5080 16GB PASS。

## 3. 自动化和恢复验收

- `py_compile`：PASS；
- 完整 `unittest discover`：43/43 PASS；
- development manifest 与 Run12 逐字节一致，SHA-256：
  `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`；
- CPU 两轮连续训练 vs 完成 epoch 0 后 SIGINT 安全恢复：完整状态最大差值 0；
- RTX 5080 + AMP + workers=2 同一恢复检查：完整状态最大差值 0；
- 比较范围覆盖 live model、EMA、optimizer、scaler、scheduler、stopper、selection、
  RNG 和下一 epoch batch order；auto resume 仅接受带 `_SUCCESS` 的完整 bundle；
- 正式 Trainer 只构造：100 epochs、patience=20、batch=16、640、workers=8、AdamW、
  AMP、device=0；optimizer 未创建，train 未调用，正式结果目录不存在。

## 4. 数据与发布边界

- development train=23,462（seg=10,483、cls=13,715、both=736）；
- development val=1,744（seg=292、cls=1,488、both=36）；
- source test 634 行全部排除；development YAML 使用 `test: null`；
- external candidates 不在 development manifest；未运行 test 或 external evaluator；
- 当前只完成工程准备，不代表正式训练已获启动授权。

## 5. 冻结标识

- config SHA-256：`e9089274a44d8c0fb352938628e11d5efc87d10621c22dc18afaf4b73ddd26d6`；
- protocol SHA-256：`6739fe54048ab52bdd789594cb11ce65de25e57a7dc5dabbfa1185b319ecaebe`；
- development data SHA-256：`894cf4a7a7d3290e6594b7fb9d7fcdfde85cdb13cc3a9ad210292be905dfa8e9`；
- entrypoint SHA-256：`f22db1eb4a1bed6587fdcc9cea87715d1884b75fd9fc85a78929bd644e26d895`；
- utility selection module SHA-256：
  `bef3ab9fc92539d3dd4607a40b888115589f93776457498db56534b139c3bcdd`；
- behavior fingerprint：
  `2433474d4f826d8fcfed94e946eb9facf54d4ac13c20fdc5e2720855ddf18e27`。
