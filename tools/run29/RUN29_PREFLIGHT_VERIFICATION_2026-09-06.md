# Run29 最终预检复核

`RUN29_PREFLIGHT_STATUS=PASS`

## 冻结身份

- config SHA-256：`540885fc8ea565287097bc8a2c2dc05d67acdba4527692ca71fde5d6ae8f0bb4`
- fingerprint：`0a4a1c81722ceacdd5143b0849739f02312e447ca0faca61a7e7b53dd338ee2e`
- 唯一因素：在 Run22 plain seed43 上增加权重 `0.5` 的恶性组层级 NLL。
- 新增参数：`0`；推理仍为 8 类 argmax；无 ECA 或 spatial attention。

## 检查结果

- 静态单元测试：`5 passed`。
- development-only GPU forward：RTX 5080，batch `16`，imgsz `640`，AMP；五项 loss 均有限，
  层级 loss 为 `0.4885508716`；未创建 optimizer、未反向、未调用 `trainer.train()`。
- formal constructor：100 epochs、batch16、imgsz640、AdamW、seed43、epoch40 后 patience30
  全部锁定；正式实验目录未创建。
- 纯合成容量预检：峰值 CUDA reserved `6996099072` bytes（约 41.0%）；一次临时 optimizer
  step 成功，未加载任何数据集。
- 纯合成中断/恢复：epoch0 安全停止后自动恢复至 epoch1；live model、EMA、optimizer、
  scaler、scheduler、stopper、selection、RNG 和第二轮 batch 顺序均无 mismatch，浮点最大差异 `0.0`。

## 数据边界

Run29 YAML 只含 development train/val 且 `test: null`。预检与结果判读均未加载或评估
IMA++ test、ISIC2018 external 或 PAD-UFES-20。

## 发布结论

`RUN29_READY_TO_CLICK_GATE=PASS`

唯一正式入口：`E:\App\DaiMa\JetBrains\Project\PiFuV2\tools\run29\train_run29_from_pycharm.py`。
由用户在 PyCharm 点击运行；Codex 不启动正式训练。

