# PiFuV2 Run15 训练前工程复核

日期：2026-08-09  
范围：train/val 静态与回归、完整恢复、真实 development batch 前向和 Trainer 构造；
未反向传播正式数据、未正式训练、未读取 test/external。

`RUN15_ENGINEERING_VERIFICATION=PASS`  
`RUN15_CPU_RESUME_EQUIVALENCE=PASS`  
`RUN15_GPU_AMP_RESUME_EQUIVALENCE=PASS`  
`RUN15_REAL_DEVELOPMENT_BATCH=PASS`  
`RUN15_FORMAL_TRAINER_CONSTRUCTOR=PASS`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`

## 1. 冻结对象

- 主基线 Run12 BLG-MDA + unweighted CE；唯一训练因素是 true-MEL direct pairwise
  softplus boundary，margin=`ln(5539/3047)=0.5976564790473948`，unit weight=1.0；
- validation/inference logits 不修改；模型参数与 Run12 初始化逐元素一致；
- config SHA-256：`efc4fe371e1c65a93ce9dddba7b45069f63732dbf3e1e555c0ec0a19e5d76f11`；
- behavior fingerprint：
  `f2e38bd9fd0422280de8c1e8ac3520d5bf6f0843efaef343d7e0601f3a43b978`；
- development manifest 与 Run12 逐字节一致，SHA-256：
  `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。

## 2. 回归与恢复

- 新 Run15 测试 8/8 PASS；项目完整回归 40/40 PASS，另 2 个 subtests PASS；
  `py_compile` PASS；
- CPU 2-epoch 连续训练与完成 epoch 1 后中断/恢复逐项等价；
- RTX 5080、Torch 2.8.0+cu129、CUDA 12.9、AMP、workers=2 同一检查 PASS；
- GPU 比较：live model 371、EMA 371、optimizer 191 个 tensor，以及 scaler、scheduler、
  stopper、selection、RNG、下一 epoch batch order 的最大绝对差均为 0；
- 自动恢复拒绝不完整或 fingerprint 不一致 bundle，只接受原子 `_SUCCESS` bundle。

## 3. 真实配置检查

- development train/val=23,462/1,744；source test 634 行排除，YAML `test: null`；
- 真实 development role batch 扩展至 batch=16、imgsz=640，以 AMP 完成 forward/loss；
- semantic/classification shape=`[16,1,80,80]`/`[16,8]`，参数量 6,614,361；
- BLG-MDA P3/P4/P5、identity ECA、mask mix=0.5、detached lesion map 均锁定；
- 该真实 role triplet 不含 MEL，pairwise item 按设计为精确 0；非零 MEL 分支已在 8 项
  公式/梯度测试和 CPU/GPU synthetic end-to-end 训练中覆盖；
- 模型 state 未改变，optimizer/backward/step/train 均未发生；
- 正式 Trainer 参数锁定 epochs100/patience20/batch16/imgsz640/workers8/AdamW/AMP/device0；
  margin、weight 和三项 selection floor 全部匹配；只构造，正式结果目录不存在。

## 4. 证据 SHA-256

- entry：`c9adb801e4bfb6782b6ed05e35002342a9bc5863a69cf96d24a14ded89ba53a9`；
- Run15 module：`c8712daef9016c4a89c9d4c1bfae2a3a824b88ad4951bc7715969f30294aeb80`；
- CPU/GPU resume reports：
  `e9e92d470359a4eee7bb430d33924ebed181374b888aea7b75d7e0c300761d3e` /
  `b0c25d92d1edbfd0520682fe5a65dad816d546217c3e364a30241070e441edec`；
- formal constructor report：
  `553c1dec5b786096240f8aca02805663374cb3361c23e7a6f1d1791f7cdaa2b4`。

## 5. 启动边界

工程门禁可以 PASS，但用户确认仍为 FALSE。正式入口必须拒绝启动，直到用户在本次
复核后明确确认。正式模型锁前禁止 test；external candidates 不参与训练或选择。
