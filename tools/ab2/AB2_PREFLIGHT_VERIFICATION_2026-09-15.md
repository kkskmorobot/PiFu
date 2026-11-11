# ab2 开训前预检核验（2026-09-15）

`AB2_PREFLIGHT_STATUS=PASS`

`FORMAL_TRAINING_STARTED=FALSE`

## 预注册与单因素边界

- 固定 seed43，与完整 LGLA lgx 和 Plain Run22 配对；
- 唯一消融是学习门控改为固定 `0.5/0.5`；
- 病灶概率注意力仍来自 `sigmoid(StopGrad(S))`；
- 门控网络为参数量匹配而保留但不参与 forward，gate-balance 正则禁用；
- test、external、PAD-UFES-20 均未进入开发配置或预检。

## 已通过检查

1. 聚焦单元测试：`6 passed`。验证固定融合精确等于 `0.5/0.5`、病灶加权局部汇聚仍有效、分类梯度不能经注意力回到语义头、保留门控参数不影响输出且无梯度、早停边界保持不变。
2. 项目回归：排除明确标记为 `superseded_run17_boundary_draft_20260810_v1` 的历史草稿后，`175 passed, 2 subtests passed`。未排除时仅在收集该归档草稿时因其旧模块缺失失败，与 ab2 无关。
3. 数据边界/prepare-only：train=`23,462`、val=`1,744`，manifest SHA-256=`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`；真实开发批次前向和 4 项损失有限，模型状态未改变，训练函数未调用。
4. CPU checkpoint save/resume：live model、EMA、optimizer、scheduler、stopper、selection、RNG 与第二 epoch 批次顺序逐项一致，最大绝对差 `0.0`。
5. CUDA/AMP/workers=2 checkpoint save/resume：同上，最大绝对差 `0.0`，AMP 检查通过。
6. RTX 5080 容量：batch=`16`、imgsz=`640`、AMP 单步前向/反向/AdamW 通过；4 项损失有限；投影梯度有限；保留门控网络梯度确认为不存在；峰值 reserved=`6,916,407,296` bytes（设备显存的 `40.5273%`）。
7. 正式 Trainer 构造：100 epochs、batch16、640、workers8、AdamW、AMP、seed43、选择合同和固定融合头全部一致；未创建优化器、未调用 `trainer.train()`、未创建正式结果目录。

## 锁定结果

- config SHA-256：`13425e3d56f405ec2d855e636c23e105a17b234bf9b55a6fee9a92572e55f637`
- fingerprint：`e50ab8dea5846fd01adfd621d1b4b025c7bbac278cec6d096e6e120d3e574738`
- 正式结果目录：不存在；正式训练进程：未启动。

结论：ab2 已到 PyCharm 可点击状态。应在 ab1 完成并释放 GPU 后再运行，二者全部完成后统一进行 validation 审计、消融比较和后续汇总。
