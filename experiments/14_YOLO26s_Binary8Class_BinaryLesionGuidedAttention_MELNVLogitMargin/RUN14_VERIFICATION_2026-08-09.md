# PiFuV2 Run14 训练前工程复核

日期：2026-08-09  
范围：仅 train/val、静态/回归、完整恢复、真实 development batch 前向和 Trainer 构造；
未训练、未反向传播、未读取 test/external。

`RUN14_ENGINEERING_VERIFICATION=PASS`  
`RUN14_FORMAL_TRAINER_CONSTRUCTOR=PASS`  
`RUN14_CPU_RESUME_EQUIVALENCE=PASS`  
`RUN14_GPU_AMP_RESUME_EQUIVALENCE=PASS`  
`RUN14_REAL_DEVELOPMENT_BATCH=PASS`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 冻结对象

- 单一因素：TMNCM；仅真实 MEL train target 的 CE 计算中给 NV competitor logit
  加 `ln(5539/3047)=0.5976564790473948`；
- validation/inference logits 不修改，无 class weight；
- fresh `models/yolo26s-sem.pt`，其余架构、数据、优化器、增强、选择规则与 Run12 不变；
- config SHA-256：`7d50195c4713656a3d1e02e0f450018dce8339cae7019a09ecdc9ebf56d31a0a`；
- behavior fingerprint：
  `30e88fffb1eab5c59d7a3a1e0cbb458dcd2b020414c1011853941ce9fd456bc4`；
- development manifest 与 Run12 逐字节一致，SHA-256：
  `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。

## 2. 回归与恢复

- Run14 新单元测试 7/7 PASS；相关完整回归 32/32 PASS（另 2 个 subtests PASS）；
  `py_compile` PASS；
- CPU 2-epoch 连续训练与 epoch 边界中断后恢复逐项等价；
- RTX 5080、Torch 2.8.0+cu129、CUDA 12.9、AMP、workers=2 的同一检查 PASS；
- GPU 比较：live model 371、EMA 371、optimizer 191 个 tensor，以及 scaler、scheduler、
  stopper、selection、RNG、下一 epoch batch order 的最大绝对差均为 0；
- 自动恢复拒绝不完整/指纹不一致 bundle，并指向最近完整 `_SUCCESS` bundle。

## 3. 真实配置检查

- development train/val=23,462/1,744；source test 634 行排除，YAML `test: null`；
- 真实 development role batch 扩展至 batch=16、imgsz=640，以 AMP 完成 forward/loss；
- semantic/classification shape=`[16,1,80,80]`/`[16,8]`；参数量 6,614,361；
- BLG-MDA 三尺度输入层 16/13/10、mask mix=0.5、lesion map detached；
- 模型 state 未改变，optimizer/backward/step/train 均未发生；
- 正式 Trainer 参数锁定为 epochs=100、patience=20、batch=16、imgsz=640、workers=8、
  AdamW、AMP、device=0；只完成构造，正式结果目录不存在。

## 4. 证据文件 SHA-256

- `train_run14_from_pycharm.py`：
  `42ddba441764e8fa065d58d54082f0d79221192ec06f95fb4259c2f6f89d68b9`；
- Run14 model/loss/trainer module：
  `9cd132796df66822139fca6ff0b981756a16bd74294974a24466e032b8010475`；
- CPU/GPU resume reports：
  `fbc8b5ac015ad26928246017e4e6ed66a2010c66ff6dd2c45cddb778c5b51249` /
  `86e8725e3ac601c22ce47c6d735f0455c227f3b2e8c80666667c4c3455040bde`；
- formal constructor report：
  `7b78536ceb54e47c29980a6d970ec227fb141016e67eb0aa22d4f34f493fd288`。

## 5. 启动边界

工程门禁可以 PASS，但用户确认仍为 FALSE。正式入口必须继续拒绝启动，直到用户在
本次复核后明确确认。反向门禁实测入口退出码为 1，唯一缺失 token 是用户确认 TRUE；
未构造 Trainer、未创建正式结果目录。正式模型锁前禁止 test；external candidates
不参与训练或选择。
