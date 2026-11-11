# 第二次尝试：YOLO11s + V2 数据平衡

- 类型：实例分割数据优化实验
- 数据：processed_skin7_v2
- 训练结果：`results\train`
- 配置：`train_yolo11s_seg_v2.yaml`
- 数据报告：`DATA_V2_REPORT.md`
- hard-example 历史缓存：`metadata\skin7_v2_hard_samples.csv`
- 论文诊断图表：`analysis`（从工作目录复制，源文件仍保留）

V2 使用更激进的类别平衡与 hard-example 增强，第 58 轮早停，整体结果下降。它是重要的负向消融证据。
