# ab1 seed43 训练前复核

日期：2026-09-15（+08:00）

## 最终判定

ab1 已准备到用户从 PyCharm 点击正式入口前一步。全部预检通过，正式训练未启动，正式结果目录
不存在；未使用 test、external 或 PAD-UFES-20 数据与指标。

## 冻结身份

- 消融：成功版 gate-balance LGLA v1 去除预测病灶空间引导；
- seed=`43`，配对完整模型=`lgx`，共同 Plain 参照=`Run22`；
- 常数全图权重使 local pooling 与 global average 精确相等，参数图和参数量保持不变；
- config SHA-256：`7a56d935766258c1be5cbab20a4e8a8ae04d94dbcaeb6267b0ef63ab4ab3e213`；
- lock fingerprint：`5bc2f059c8d85d45aa6addd4496ad706ca7de03144cf9fe28d9aa0ad956ce122`。

## PASS 证据

1. 全部项目回归：`169 passed, 2 subtests passed in 5.79s`。
2. CPU checkpoint/save/resume：
   `preflight/ab1_checkpoint_resume_v1/cpu_20260915_v1/PREFLIGHT_REPORT.json`；全部状态最大差值 0。
3. CUDA/AMP checkpoint/save/resume：
   `preflight/ab1_checkpoint_resume_v1/gpu_amp_workers2_20260915_v1/PREFLIGHT_REPORT.json`；全部状态最大差值 0。
4. batch=16、imgsz=640、AMP 容量：
   `preflight/ab1_yolo26m_capacity_v1/batch16_img640_amp_20260915_v1.json`；5 项 loss 有限，
   backward 与 AdamW 单步通过，峰值 reserved=6,916,407,296 bytes（40.527%）。
5. prepare-only：`tools/ab1/AB1_PREFLIGHT_REPORT.json`；真实 development role batch 前向/loss
   通过，uniform mass fraction 全为 1，参数量 14,540,748，未创建 optimizer，未训练。
6. 正式 Trainer 构造：
   `preflight/ab1_formal_trainer_constructor_v1/lgla_v1_unguided_gate_balance_constructor_20260915_v1/CONSTRUCTOR_REPORT.json`；
   seed43、batch16、100 epochs、AdamW、AMP、gate-balance 0.01 正确；未创建 optimizer，未调用 train。

## 数据边界

- train：total 23,462；segmentation 10,483；classification 13,715；both 736；
- validation：total 1,744；segmentation 292；classification 1,488；both 36；
- manifest 与 Run22 development manifest SHA-256 一致；YAML `test=null`，无 external；
- IMA++ test、ISIC2018 external、PAD-UFES-20 均未访问。

## 停止点

唯一正式入口：`tools/ab1/train_ab1_from_pycharm.py`。当前正式目录
`experiments/ab1_YOLO26m_Binary8Class_LGLAv1_Unguided_GateBalance_Seed43/results/ab1_yolo26m_lgla_v1_unguided_gate_balance_seed43`
不存在。下一步只能由用户在 PyCharm 手动运行；Codex 未启动正式长训练。
