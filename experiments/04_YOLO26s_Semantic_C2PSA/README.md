# 第四次尝试：YOLO26s Semantic + C2PSA

- 类型：语义分割与原生注意力基线
- 数据：processed_skin7_v3
- 训练与分析结果：`results`
- 训练配置：`train_yolo26s_sem_v4.yaml`
- 修正后的独立评价配置：`data_semantic_eval_v4.yaml`

训练在第 91 轮正常早停，语义 best 对应第 61 轮，mIoU=0.59122。图像级 Macro-F1=0.7041。该实验作为后续自定义注意力消融的 YOLO26 原生 C2PSA 基线。

注意：原始 7 类 YAML 直接进行 standalone semantic validation 会错误地把背景映射到 vasc；独立验证必须使用显式 background=7 的 8 类配置。
