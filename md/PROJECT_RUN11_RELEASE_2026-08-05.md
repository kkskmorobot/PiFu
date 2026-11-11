# PiFuV2 Run11 最终放行决议

更新时间：2026-08-05  
项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`RUN10_RESULT_REGISTRATION_GATE=PASS`  
`RUN10_PRESERVATION_GATE=PASS`  
`RUN11_SINGLE_FACTOR_PREREGISTRATION_GATE=PASS`  
`RUN11_MEL_WEIGHT_IMPLEMENTATION_GATE=PASS`  
`RUN11_PARTIAL_LABEL_REGRESSION_GATE=PASS`  
`RUN11_CHECKPOINT_CONTINUOUS_RESUME_GATE=PASS`  
`RUN11_GPU_FORWARD_PREFLIGHT_GATE=PASS`  
`RUN11_FORMAL_TRAINER_CONSTRUCTOR_GATE=PASS`  
`RUN11_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`PROJECT_RUN11_GATE=PASS`

用户已明确允许写入 PiFuV2 的 Run11 新文件，并要求在全部门禁通过后使用 PyCharm
启动正式训练。本决议放行唯一冻结入口的用户可见启动；不放行 test 或 external。

## 1. Run10 登记与保护

Run10 已完成 100 个连续 epoch，训练与断点完整，但 100 个 epoch 中没有任何状态
满足 `mel->nv <= 0.1077844311`，故按预注册协议登记为约束失败且不创建正式模型锁。
Run10 结果目录保持 44 个文件、820,603,619 bytes：

- `results.csv` SHA-256：
  `f859e0b8632e79c3b97fe5242fb706f65919e48fc20b6726eff72a892331db8d`；
- `RUN10_TRAINING_COMPLETE.json` SHA-256：
  `46d1f8e124c0e7be01629117073975e19736f8950939d1aef69cb3374c4742b3`。

## 2. Run11 唯一因素

相对 Run10，只改变分类 CE 的真实 MEL target 权重：

`w_mel = N_train(nv) / N_train(mel) = 5539 / 3047 = 1.8178536265178864`

其余七类为 1.0。权重只由 development train 行得到；没有使用 val、test 或
external 调参。Run11 从与 Run10 相同的 `models/yolo26s-sem.pt`、seed 42 fresh
start，不从 Run10 checkpoint 续训。数据、模型头、采样、增强、优化器、100 epoch
预算、partial-label mask、指标、风险阈值与选择规则全部不变。

## 3. 冻结文件

- 配置：`experiments\11_YOLO26s_Binary8Class_MELWeightedCE\run11_preregistered_config_v1.yaml`
  - SHA-256：`3ed63e0ac9ab6993eb293c8dbd6c02aa5df4fb1ee5eb84440448ab7d4c1b2bb4`
- 协议：`experiments\11_YOLO26s_Binary8Class_MELWeightedCE\RUN11_PROTOCOL_2026-08-05.md`
  - SHA-256：`27b6b5c3cff506e722ad10f7c54b91b0744f09f753623b1173020674df8a6945`
- 模型/损失：`tools\imaplusplus_partial_multitask_mel_weight_v1.py`
  - SHA-256：`6a4f4aeb614426f67817cab9378fb86ce76a4f4a26a67c25b846050187d8c53a`
- PyCharm 入口：`train_run11_from_pycharm.py`
  - SHA-256：`ee1ad4c0d4d5048af701d090bbb6f9fa880ba2ce7899b297641fb5ba9e86c148`
- behavior/data/code fingerprint：
  `de4e5dc5fcc6e7facf5cc17e1b3453600016e92a3fd8a8b66147b149709cb068`

Run11 development manifest 与 Run10 的 train/val 投影逐字节一致，SHA-256 为
`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`；
Run11 YAML 的 test 为 null，PAD/ISIC2018 external 不在入口中。

## 4. 预检证据

- 12 个 partial-label、MEL weighted CE、checkpoint 单元测试全部 PASS；
- RTX 5080、PyTorch 2.8.0+cu129、CUDA 12.9；
- 正式 batch 16、640、AMP 的真实 development forward/loss PASS；
- MEL-only class weight 向量验证 PASS；peak allocated 593,635,328 bytes；
- model state 未改变，没有 optimizer/backward/step/train 调用；
- CPU 连续 vs 中断恢复：全部状态和第二轮 batch 顺序零差异；
- RTX 5080 AMP + workers=2 连续 vs 中断恢复：零差异、failures=0；
- 正式 Trainer overrides 构造 PASS，正式 results 目录在启动前不存在。

## 5. 正式入口和恢复

PyCharm 只运行：

`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run11_from_pycharm.py`

解释器为项目 `.venv\Scripts\python.exe`，Parameters 留空，working directory 为项目
根目录。首次为 fresh start；安全中断后再次运行同一入口会 auto resume 最新完整、
哈希和 fingerprint 匹配的 `_SUCCESS` bundle。入口没有 test/external 参数。

训练完成或正常早停后返回 Codex 做 val-only 锁定；在新的用户明确授权前，禁止运行
IMA++ test、ISIC2018 external 或 PAD external。
