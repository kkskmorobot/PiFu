# PiFuV2 Run11 预注册协议

日期：2026-08-05  
基线：`experiments/10_YOLO26s_Binary8Class_PartialLabel_Baseline`  
实验目录：`experiments/11_YOLO26s_Binary8Class_MELWeightedCE`

## 1. Run10 触发依据

Run10 完成连续 100 个 epoch，`results.csv` 无缺行或非有限值，完整 bundle
保留且可恢复。Run10 的无约束最佳联合状态位于日志 epoch 45：binary IoU
0.78113、group Macro-F1 0.59426、joint score 0.68132，但 mel->nv rate
为 0.43878。100 个 epoch 中没有任何状态满足冻结的
`mel->nv <= 0.1077844311`，因此 Run10 是完整训练后的约束失败基线，不能借助
test 或 external 结果晋升模型。

## 2. 唯一变更因素

Run11 只改变图像分类交叉熵中的真实 MEL 样本权重：

`w_mel = N_train(nv) / N_train(mel) = 5539 / 3047 = 1.8178536265178864`

其余七类权重均为 1.0。该常数只由冻结 development manifest 的 train 行计算；
没有使用 val、test、ISIC 2018 或 PAD-UFES-20 选择权重。目标是抵消 NV 相对 MEL
的训练曝光差异，并直接检验这一损失代价是否降低 mel->nv。

## 3. 严格保持不变

- IMA++/ISIC 2019 train/val 行、partial-label manifest 投影、split 和标签语义；
- 23,462 train / 1,744 val，每轮标准有种子无放回 shuffle；
- YOLO26s binary semantic backbone、P3/P4 plain classification head 和 dropout 0.20；
- `models/yolo26s-sem.pt` 初始化权重；Run11 不从 Run10 权重续训；
- image classification loss 总权重 0.5，label smoothing 0；
- batch 16、imgsz 640、100 epochs、patience 20、AdamW、学习率、增强、seed 42、AMP；
- binary segmentation loss、partial-label masking、validator 和全部 metric 定义；
- 每 epoch 完整 checkpoint bundle v2、safe signal boundary、auto/explicit resume；
- test/external 禁止进入训练、选择、阈值或权重调整。

除 MEL target CE 权重外，不引入 class-balanced sampler、全类 inverse-frequency
weight、focal/asymmetric loss、margin、attention、label smoothing、temperature fitting
或 threshold tuning。

## 4. 冻结 val-only 选择

- `joint_score = sqrt(binary_iou * group_macro_f1)`；
- feasible iff val group mel->nv rate <= 0.1077844311 且 MEL support 存在；
- primary alias：`best_feasible_joint`；
- tie：更高 group Macro-F1、再更高 binary IoU、再更早 epoch；
- semantic threshold=0.5，classification=argmax，temperature=1.0；
- 若仍无 feasible checkpoint，Run11 继续判约束失败；无约束最佳状态仅作诊断。

必须同时报告 binary IoU/Dice、image/group Macro-F1/Recall、mel->nv、nv->mel、
malignant->benign、NLL、Brier、ECE 和 joint score。训练过程中不得修改约束。

## 5. 完整恢复合同

每个完成 epoch 原子发布 `_SUCCESS` bundle，包含 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、stopper、selection state、Python/NumPy/Torch CPU/CUDA RNG、
sampler/data progress、manifest SHA-256 和 behavior fingerprint。`resume=auto` 只选择
最新完整且指纹一致的 bundle；SIGINT/SIGTERM 仅在完成 epoch 的安全边界停止。

## 6. 启动与边界

正式训练只能由用户在 PyCharm 运行项目根目录的
`train_run11_from_pycharm.py`。默认无参数：首次 fresh start，之后 auto resume。
Codex 的 `--prepare-only` 只能做数据、CUDA、模型 forward/loss 和门禁检查，不能
创建 optimizer、backward、step 或调用 `trainer.train()`。

Run11 结束后先返回 Codex 做 val-only 锁定。未形成正式模型锁且未获得用户新的
明确授权前，不得运行 IMA++ test、ISIC2018 external 或 PAD external。
