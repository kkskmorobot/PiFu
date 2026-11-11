# Run17 预注册协议：YOLO26m 无注意力多任务基线

`RUN17_PROTOCOL_STATUS=PREREGISTERED`

## 研究目的

Run16 已完成 YOLO26m + 二值病灶引导多尺度注意力（BLG-MDA），但此前的无模块基线 Run10 使用 YOLO26s，不能作为主干更换后的严格注意力消融对照。Run17 补齐同一 YOLO26m 主干下的无注意力基线。

较早生成的 Run17 边界注意力方案只完成过草案及预检，从未正式训练。其文件在本协议生效前完整归档，旧 ready marker 作废；正式 Run17 以本协议定义的无注意力基线为唯一有效合同。边界注意力变体暂缓且不计入训练结果。

## 唯一预注册因素

相对 Run16，唯一变化是删除图像分类头中的二值病灶引导注意力：

- 删除 P3/P4/P5 的 ECA 通道门控；
- 删除由分割 logit 生成的 detached 病灶空间权重；
- 删除 P5 图像描述符；
- 分类头恢复为基础实现的 P3/P4 全局平均池化 + MLP。

分割主干、分割头、预训练权重、数据、split、partial-label 规则、分类 taxonomy、损失权重、采样、优化器、学习率、增广、batch、分辨率、epoch、patience、AMP、seed、统一 val 指标、可行性阈值和 checkpoint 合同全部保持 Run16 不变。

## 论文对比

主要对照是 Run17（YOLO26m，无注意力）与 Run16（YOLO26m，BLG-MDA）。两者均为 seed=42 的单次受控实验。只有在该对比完成后，才决定是否进行额外注意力变体或多随机种子确认。

## 数据边界

- train/val 仅使用冻结的 development manifest；
- IMA++ source test 634 对完全排除；
- ISIC 2018 Task 1 incremental 与 PAD-UFES-20 仅是外部候选，不参与训练、选择或阈值拟合；
- 正式模型锁定和用户新授权前禁止运行 test 或外部评价。

## 训练与恢复

- PyCharm 入口：`train_run17_from_pycharm.py`；
- fresh start 使用 `models/yolo26m-sem.pt`；
- batch=16，imgsz=640，epochs=100，patience=20，AdamW，AMP，seed=42；
- 默认 `resume=auto`，只接受通过大小、SHA-256、指纹和 `_SUCCESS` 校验的完整 epoch bundle；
- bundle 保存 live model、EMA、optimizer、scaler、scheduler、stopper、selection、Python/NumPy/Torch CPU/CUDA RNG 和数据顺序状态。

## 停止规则

Run17 训练完成或正常早停后，只做统一 val 评价并与 Run16 比较。若没有正式完成标记，不进行下一实验。Run17 本身不运行 test 或外部评价。
