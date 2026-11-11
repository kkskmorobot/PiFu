# Run20 预注册协议：自适应残差病灶引导注意力（AR-BLG-MDA）

`RUN20_PROTOCOL_STATUS=PREREGISTERED`

## 依据

Run19 相对 Run18 在 epoch 1–39 的平均 joint 高 0.01051，但在 epoch 40–71 低 0.00657，后期 group Macro-F1 低 0.01791。现有 BLG-MDA 从第一轮开始固定使用 0.5 的病灶引导池化比例，而且三个尺度共享同一比例；当分割预测尚未成熟或不同尺度需要不同上下文时，分类头无法降低该引导强度。

## 唯一预注册因素

相对 Run19，只改变病灶引导池化的混合系数：

- 固定共享 `mask_guided_mix=0.5` 改为 P3/P4/P5 三个可学习参数；
- 每个实际 mix=`0.5*sigmoid(logit)`，严格限制在 `(0,0.5)`；
- 三个尺度均从 0.05 初始化，使模型先接近全局池化，再由分类损失学习是否及在何种尺度增强病灶聚焦；
- 池化写为 `global + mix*(guided-global)`，作为全局描述符上的残差注意力。

P3/P4/P5、identity-initialized ECA、detached 分割概率图、P5 初始零贡献、YOLO26m 主干、数据、损失、采样、优化器、增广、batch=16、imgsz=640、100 epochs、早停、统一 val、风险约束和 checkpoint 合同均保持 Run19 不变。

## 数据和测试边界

train/val 只使用冻结 development manifest；IMA++ test、ISIC 2018 外部候选和 PAD-UFES-20 均不参与训练、选择或阈值拟合。正式模型锁定并取得新授权前禁止 test 或外部评价。

## 训练入口

唯一入口为 `tools/run20/train_run20_from_pycharm.py`。默认 `resume=auto`；最早可见 epoch 70 早停，最多 100 epochs。

## 判断标准

Run20 首先与 Run19 比较注意力机制改进是否减少后期退化，同时以 Run18 普通头作为参考。没有 feasible checkpoint 不等同于注意力失败；MEL→NV 风险问题仍单独报告，不在本轮混入损失或采样修改。
