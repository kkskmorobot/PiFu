# Run32 正式发布（2026-09-08）

`PROJECT_RUN32_GATE=PASS`

`RUN32_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 发布对象

- 唯一入口：`tools/run32/train_run32_from_pycharm.py`
- 配置 SHA-256：`221cb238e15bacc0d1c1a6a111d7f800aa0a93ae28aeebd238b76fb0b54c71a1`
- Run32 fingerprint：`df95baaec5fde165c600f8ace75cc15c0ca4ef4076aff8f899d51b050da1862c`
- 正式结果目录：`experiments/32_YOLO26m_Binary8Class_Plain_MalignantAuxHead_Seed43/results/run32_yolo26m_plain_malignant_aux_head_seed43`

## 单一因素

相对同 seed43 的冻结 Run22 plain 基线，只增加训练期恶性/良性二类辅助头：从已有 256 维
共享隐藏描述符接出 `Linear(256,2)`，新增 514 个参数；全部官方分类训练行按冻结 taxonomy
映射为 benign=`[0,2,3,5,6]`、malignant=`[1,4,7]`，辅助权重固定为 `0.25`。
主 8 类 logits、验证指标、checkpoint 选择及 eval/inference argmax 不变。

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

用户已明确要求依据 Run31 结果选择下一实验，并准备到其本人点击训练之前。Codex 不得替用户启动；
用户须在 PyCharm 中打开唯一入口，确认 Run32 配置后点击 Run。入口默认 `resume=auto`，首次运行因
正式结果目录不存在而 fresh start。
