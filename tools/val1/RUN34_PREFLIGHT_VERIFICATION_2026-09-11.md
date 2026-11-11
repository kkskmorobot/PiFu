# Run34 预检复核（2026-09-11）

`RUN34_PREFLIGHT_VERIFICATION_STATUS=PASS`

Run34 是 development-only 冻结模型决策评估，不是训练 run。因此不需要训练专用的 GPU 容量、optimizer 或中断恢复预检，但必须完成以下评估专用门：

- 入口与单元测试静态通过；
- Run18/22/24 bundle 大小、SHA-256、`_SUCCESS` 与 fingerprint 通过；
- Run34 自有 development YAML/manifest 与冻结 taxonomy 通过；
- 用 `trainer.pt` 恢复结构，再加载 `runtime_state.pt` 中的无损 fp32 EMA；
- 使用未融合 Conv+BN、CUDA autocast 的训练期验证数值路径；
- Run18 完整 validation argmax 指标复现误差不超过 `5e-4`；
- 不创建 optimizer，不调用 backward/train，不访问 test/external/PAD-UFES-20。

最终复核结果：

- Python 编译通过，9 项单元测试通过；
- Run18 完整 validation 复现通过；
- Binary IoU 差值 `0.0`；
- Image Macro-F1 差值 `-4.68e-06`；
- Group Macro-F1 差值 `+4.37e-06`；
- Group Recall 差值 `-2.14e-06`；
- MEL→NV 差值 `-4.90e-06`；
- malignant→benign 差值 `+2.32e-06`；
- Joint 差值 `-1.81e-06`。

所有复现差值均远小于预注册容差 `5e-4`。
