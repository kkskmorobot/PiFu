# lg2 seed42 训练前复核

日期：2026-09-13（+08:00）

## 最终判定

lg2 已准备到用户从 PyCharm 点击正式入口前一步。全部授权预检通过，正式训练未启动，正式 result
目录不存在；未使用 test、external 或 PAD-UFES-20 数据与指标。

## 冻结身份

- 结构：lgx 已通过筛选门的 gate-balanced LGLA v1，行为代码仅将实验标识从 lgx 改为 lg2；
- seed=`42`，配对 Plain=`Run18`；
- config SHA-256：`ef019674fe95ae80560285abfc74fe046b72f4b3a473fb05334cae4aef19f170`；
- lock fingerprint：`16f9505ad452813aac9eb85e1b4cb176d077ff9d16a30ac20965ff3efbd87df4`；
- development manifest SHA-256：`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。

## PASS 证据

1. 当前正式测试与保留回归测试：`155 passed, 2 subtests passed in 6.38s`。
2. CPU checkpoint/save/resume：`preflight/lg2_checkpoint_resume_v1/20260913_203945/PREFLIGHT_REPORT.json`；
   live model、EMA、optimizer、RNG 与第二 epoch 顺序在容差 0 下完全一致。
3. CUDA/AMP checkpoint/save/resume：`preflight/lg2_checkpoint_resume_v1/20260913_204015/PREFLIGHT_REPORT.json`；
   RTX 5080 上所有比较状态最大差值为 0，低于 1e-6。
4. YOLO26m batch=16、imgsz=640、AMP 容量预检：
   `preflight/lg2_yolo26m_capacity_v1/batch16_img640_amp_20260913_v1.json`；5 项 loss 有限，完整
   backward 与 AdamW step 通过，峰值 reserved=6,914,310,144 bytes（40.515% 显存）。
5. 最终 prepare-only：`tools/lg2/LG2_PREFLIGHT_REPORT.json`；真实 development role batch 前向/loss
   通过，参数 14,540,748，optimizer/backward/trainer.train 均未调用。
6. Ready marker：`tools/lg2/LG2_READY_TO_TRAIN.json`；status=PASS、ready_to_click_pycharm=true。
7. 正式 Trainer 构造：
   `preflight/lg2_formal_trainer_constructor_v1/lgla_v1_gate_balance_constructor_20260913_v1/CONSTRUCTOR_REPORT.json`；
   seed42、batch16、100 epochs、AdamW、AMP、gate balance 0.01 与 class 均正确；未创建 optimizer，
   未调用 trainer.train()。

## 数据边界

- train：total 23,462；segmentation 10,483；classification 13,715；both 736；
- validation：total 1,744；segmentation 292；classification 1,488；both 36；
- manifest 与冻结 Run22 development manifest 字节一致；YAML test=null；无 external candidate；
- `imaplusplus_test=false`、`isic2018_external=false`、`pad_ufes_20=false`。

## 停止点

唯一正式入口：`tools/lg2/train_lg2_from_pycharm.py`。当前
`experiments/lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42/results/lg2_yolo26m_lgla_v1_gate_balance_seed42`
不存在。下一步只能由用户在 PyCharm 手动运行该文件；Codex 未启动正式长训练。
