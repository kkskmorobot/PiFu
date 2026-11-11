# Run30 最终预检复核

`RUN30_PREFLIGHT_STATUS=PASS`

## 冻结身份

- config SHA-256：`cae5b6571e724529c36ea92fbff4dca73a27002130232db2343d4931e0179ce0`
- fingerprint：`aa8336f13f4c3b024eb5eb09b7bb00b7311d639dad6d9ff4196b3ec61213f9e4`
- 唯一因素：Run22 plain seed43 的 8 类 CE 增加权重 `0.5`、gamma `2.0` 的
  true-MEL→NV posterior focal risk。
- 新增参数 `0`，推理规则不变，无注意力模块。

## 检查结果

- 静态单元测试：`6 passed`；公式、true-MEL 归一化与非 MEL 零梯度通过。
- development-only GPU forward：RTX 5080、batch16、imgsz640、AMP；五项 loss 有限，
  原始 role batch 不含 MEL，因此真实 batch 辅助项为 0；独立 GPU true-MEL 风险探针
  `0.1900124997`，确认公式已执行。模型状态未改变，未创建 optimizer，未调用训练。
- formal constructor：100 epochs、batch16、imgsz640、AdamW、seed43、epoch40 后
  patience30、plain 分类头和风险参数全部锁定；正式实验目录未创建。
- 纯合成容量：一次临时 batch16 optimizer step 成功；峰值 CUDA reserved
  `6996099072` bytes（约 41.0%）。
- 纯合成恢复：epoch0 安全中断后恢复至 epoch1；live model、EMA、optimizer、RNG、
  stopper、selection 与第二轮 batch 顺序均无 mismatch，最大浮点差异 `0.0`。

## 数据边界与发布结论

Run30 仅引用 development train/val 且 `test: null`。未加载或评估 IMA++ test、
ISIC2018 external 或 PAD-UFES-20。

`RUN30_READY_TO_CLICK_GATE=PASS`

唯一正式入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\tools\run30\train_run30_from_pycharm.py`。
由用户在 PyCharm 点击；Codex 不启动正式训练。

