# ab2 正式训练放行（2026-09-15）

`PROJECT_AB2_GATE=PASS`

`AB2_READY_TO_CLICK_GATE=PASS`

`USER_CONFIRMATION_RECORDED=TRUE`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## 冻结实验

- 唯一因素：完整 LGLA 的学习型两路 softmax 门控替换为固定 `0.5/0.5` 全局/病灶局部融合；
- 预测病灶注意力、StopGrad、P3/P4、共享投影、分类器、数据、增强、优化器、早停、选择和 checkpoint 合同保持不变；
- 门控 MLP 保留在模型和 checkpoint 中以保持参数量一致，但 forward 不调用、无梯度；
- gate-balance 正则随固定门控禁用，锁定权重为 `0.0`；
- seed=`43`，epochs=`100`，batch=`16`，imgsz=`640`，workers=`8`，AdamW，AMP。

## 锁定标识

- config SHA-256：`13425e3d56f405ec2d855e636c23e105a17b234bf9b55a6fee9a92572e55f637`
- fingerprint：`e50ab8dea5846fd01adfd621d1b4b025c7bbac278cec6d096e6e120d3e574738`
- 参数量：`14,540,748`，与完整 LGLA 一致。

## 启动边界

唯一入口：`tools\ab2\train_ab2_from_pycharm.py`，无参数运行；默认 `resume=auto`。

正式结果目录在放行时不存在，AI 未调用正式训练路径。用户应在 ab1 完成并释放 GPU 后再启动 ab2，不要让两次长训练在同一张 GPU 上同时运行。
