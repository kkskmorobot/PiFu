# Run17 边界注意力草案状态

`RUN17_BOUNDARY_DRAFT_STATUS=SUPERSEDED_BEFORE_FORMAL_TRAINING`

该目录对应的边界注意力方案只完成了开发预检，从未创建正式 results 目录，也未启动训练。为补齐毕业论文在 YOLO26m 主干下的无注意力基线，正式 Run17 已重新预注册为：

`experiments/17_YOLO26m_Binary8Class_PartialLabel_Baseline`

旧草案的入口、配置、协议、manifest、ready/preflight、模型模块、测试和恢复/构造器预检已复制归档至：

`preflight/superseded_run17_boundary_draft_20260810_v1`

本目录中的旧 ready marker 已作废，禁止据此启动训练；边界注意力变体如需训练，必须使用后续新实验编号重新预注册。
