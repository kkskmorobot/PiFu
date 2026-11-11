# Run13 实现与训练前验收

日期：2026-08-08  
实验：`experiments/13_YOLO26s_Binary8Class_BinaryLesionGuidedAttention_MELWeightedCE`

`RUN13_IMPLEMENTATION_GATE=PASS`  
`RUN13_REGRESSION_TEST_GATE=PASS`  
`RUN13_CPU_RESUME_GATE=PASS`  
`RUN13_GPU_RESUME_GATE=PASS`  
`RUN13_FORMAL_CONSTRUCTOR_GATE=PASS`  
`RUN13_REAL_BATCH_FORWARD_GATE=PASS`  
`RUN13_TEST_DATA_GATE=NOT_TOUCHED`  
`RUN13_EXTERNAL_DATA_GATE=NOT_TOUCHED`  
`RUN13_FORMAL_TRAINING_STARTED=FALSE`

## 1. 单因素与初始化等价性

- 相对 Run12 唯一变化：分类 CE 的 MEL target weight 从 1.0 改为
  `5539/3047=1.8178536265178864`；
- Run13 模型 state dict 与 Run12 在 seed=42 下逐元素一致；
- 参数量同为 6,614,361，classification head 参数 115,848，17.4 GFLOPs；
- P3/P4/P5 source indices `[16,13,10]`、channels `[128,256,512]`、3 个
  identity-initialized ECA、detached lesion map、mask mix=0.5 全部不变；
- pretrained transfer 360/371 items；
- 非 MEL-only batch 的 weighted CE 与 unweighted CE 相同；混合 MEL batch 与 PyTorch
  冻结权重参考公式一致。

## 2. 自动测试

- `py_compile`：PASS；
- Run10~Run13 partial-label/attention/MEL/checkpoint 回归：25/25 PASS；
- 其中 Run13 新增 6 项：精确权重、权重漂移拒绝、Run12 初始化等价、criterion 类型、
  PyTorch 参考 CE、非 MEL batch 等价。

## 3. 完整断点恢复

- CPU、workers=0、无 AMP：PASS；
- RTX 5080、AMP、workers=2：PASS；
- 连续两 epoch 与安全中断后 auto-resume 的 live model、EMA、optimizer、scaler、
  scheduler、stopper、selection、RNG、下一 epoch batch order 最大差值均为 0；
- SIGINT 在完成 epoch 边界保存 `_SUCCESS` bundle；auto-resume 指向最新完整兼容 bundle；
- GPU 环境：NVIDIA GeForce RTX 5080，Torch 2.8.0+cu129，CUDA 12.9。

CPU 报告：`preflight/run13_checkpoint_resume_v1/cpu_20260808_v1/PREFLIGHT_REPORT.json`  
GPU 报告：`preflight/run13_checkpoint_resume_v1/gpu_amp_workers2_20260808_v1/PREFLIGHT_REPORT.json`

## 4. 正式构造与真实 batch

- 正式 Trainer：epochs=100、patience=20、batch=16、imgsz=640、workers=8、AdamW、
  AMP、device=`"0"`；
- MEL 权重和 BLG-MDA 配置逐项匹配；未创建 optimizer，未调用 train；
- 第一次 constructor v1 仅因测试器预期整数 device=0、而 Ultralytics 规范化为字符串
  `"0"` 失败；没有训练副作用。修正测试期望后的 constructor v2 PASS；
- 真实 development role batch=16、imgsz=640、AMP forward/loss PASS；
- forward loss items：`[0.65644896, 0.60189843, 0.27448013, 2.05379963]`；
- GPU peak allocated/reserved：600,452,608 / 1,004,535,808 bytes；
- model state 未改变，未 backward、未 optimizer step、未调用 trainer.train。

## 5. 数据边界

- development train 23,462（seg 10,483；cls 13,715；both 736）；
- development val 1,744（seg 292；cls 1,488；both 36）；
- Run13 manifest 与 Run12 逐字节一致；
- source test 634 行排除，YAML `test: null`；
- external candidates 不在 development manifest；PAD 2,298 PNG 只核验解压完整性，
  training role 仍为 `none_external_candidate_only`。

## 6. 冻结标识

- config：`269b58b3e5062a3f9b279f6d60365c0f5f25d846f656fbb72e10e66dea008674`；
- protocol：`82e3e89c1f9f98f3c58ab9c3ab3e6a8f97d61da2c3c076f17ee12a08a15c4115`；
- development manifest：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`；
- entrypoint：`378b4969b8d0165f19913536e448f0e264bc68523e74d450827f65695af46ff6`；
- combination module：`2018be48f3309b37bfed567667ba5728321976ff346fbf0446a1d845cc5b7465`；
- final preflight report：`98a9104a0fd9d75aa04f04351b7ffc222ff5a2073112d31a3d29c2c7adefbacc`；
- ready marker：`19a3004a714154accd3f154f36cb17bc87fa7501fdcc55ab66de7d5729942112`；
- behavior fingerprint：`980198023b0119e63882eb5d5f5fedf662691eb4837a775c5b73b0696d38375b`。

以上证明 Run13 达到“等待用户确认后可从 PyCharm 点击”的状态，不代表正式训练、
正式模型选择或 test 授权已经发生。
