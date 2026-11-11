# 第三次尝试：YOLO11s + V3 NV3200

- 类型：实例分割稳定数据基线
- 数据：processed_skin7_v3
- 训练结果：`results\train`
- 配置：`train_yolo11s_seg_v3.yaml`
- 数据报告：`DATA_V3_REPORT.md`
- 图像级诊断、混淆矩阵与典型误判：`analysis`（从工作目录复制，源文件仍保留）

V3 只把 V1 的训练 NV 从 3500 调整到 3200，其他主要变量保持一致。该实验是当前 YOLO11s 稳定对照。
