# Run31 正式发布（2026-09-07）

`PROJECT_RUN31_GATE=PASS`

`RUN31_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 发布对象

- 唯一入口：`tools/run31/train_run31_from_pycharm.py`
- 配置 SHA-256：`69168bc82fde8b01a092cd48ce6cdb6c4a4e58b56f29edaa3c75886010c82378`
- Run31 fingerprint：`6200795f73c0adee421b1d0e0086fdfd432e314017e0c03c8465af47fc8ddc69`
- 正式结果目录：`experiments/31_YOLO26m_Binary8Class_Plain_MELNVAuxHead_Seed43/results/run31_yolo26m_plain_mel_nv_aux_head_seed43`

## 单一因素

相对同 seed43 的冻结 Run22 plain 基线，只增加训练期 MEL/NV 二类辅助头：从已有 256 维
共享隐藏描述符接出 `Linear(256,2)`，新增 514 个参数；仅对官方 MEL/NV 训练行计算 CE，
辅助权重固定为 0.5。主 8 类 logits、验证指标、checkpoint 选择及 eval/inference argmax 不变。

## 通过的发布门

- 单元测试：6/6 PASS。
- `--prepare-only`：PASS；真实 development role batch16 前向与损失有限；未创建优化器、未反向、
  未调用 `trainer.train()`。
- 正式 Trainer 构造：PASS；seed43、batch16、imgsz640、AdamW、AMP、early-stop 与选择协议锁定。
- RTX 5080 合成容量：PASS；batch16/img640/AMP 完成一次前向、反向与 AdamW step；总参数
  `14,507,852`，新增辅助参数 `514`。
- checkpoint 中断/恢复：PASS；live/EMA/optimizer/RNG/第二 epoch 顺序在容差 `1e-6` 下零差异。
- development manifest 仅含 train/val；YAML `test: null`；未使用 IMA++ test、ISIC2018
  external 或 PAD-UFES-20。

## 启动授权边界

用户已明确要求准备到点击训练前一步。Codex 不得替用户启动；用户须在 PyCharm 中打开唯一入口，
确认 Run31 配置后点击 Run。入口默认 `resume=auto`，首次运行因正式结果目录不存在而 fresh start。
