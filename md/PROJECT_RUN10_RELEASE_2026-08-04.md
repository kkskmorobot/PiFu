# PiFuV2 Run10 最终放行决议

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`CLASSIFICATION_TAXONOMY_GATE=PASS`  
`CLASSIFICATION_INCREMENTAL_DATASET_GATE=PASS`  
`PARTIAL_LABEL_LOADER_GATE=PASS`  
`MULTITASK_MODEL_IMPLEMENTATION_GATE=PASS`  
`UNIFIED_VAL_PROTOCOL_IMPLEMENTATION_GATE=PASS`  
`CHECKPOINT_BUNDLE_CONTRACT_V2_GATE=PASS`  
`CHECKPOINT_CONTINUOUS_RESUME_EQUIVALENCE_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=PASS`  
`RUN10_PREREGISTRATION_GATE=PASS`  
`RUN10_FINAL_PREFLIGHT_GATE=PASS`  
`RUN10_READY_TO_CLICK_GATE=PASS`  
`USER_CONFIRMATION_RECORDED=TRUE`  
`FORMAL_TRAINING_STARTED=FALSE`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`PROJECT_RUN10_GATE=PASS`

这是当前最新 Run10 决议。用户在 Stage 3 BLOCKED 文档之后明确要求：操作到
点击训练前的最后一步，并告知点击哪个脚本。因此用户确认已记录；本决议放行
Run10 的冻结设计与用户手动启动，不代表已经启动训练，也不放行 test/external
evaluation。

## 1. 唯一正式入口

PyCharm 中只运行：

`E:\App\DaiMa\JetBrains\Project\PiFuV2\train_run10_from_pycharm.py`

- 使用项目解释器：`E:\App\DaiMa\JetBrains\Project\PiFuV2\.venv\Scripts\python.exe`。
- 无需填写 Parameters；默认 `resume=auto`。
- 第一次点击时 results 不存在，入口开始全新 Run10。
- SIGINT/SIGTERM 后会在完整 epoch 边界保存并停止；再次点击同一脚本会自动恢复
  最新完整兼容 bundle。
- 若在首个完整 epoch 前异常退出，入口拒绝覆盖不完整目录，必须先审计。
- 入口没有 test/external 评价参数。

入口 SHA-256：
`7b9e7e61716820761d9333db121e1949392908be0b140213caa0a132dacb6aa9`。

## 2. 冻结 Run10 基线

Run10 是新 binary segmentation + official 8-class partial-label 合同的首个正式
基线，不宣传为与 Run9 的因果单因素对比。登记的单一合同迁移是：

`migrate_to_binary_segmentation_plus_official_8class_partial_labels_v1`

合同内部不加入注意力、lesion/class sampler、class weight、focal/asymmetric loss、
label smoothing 或 calibration fitting。后续每个新编号实验只能相对 Run10 改一个
预注册因素。

冻结配置：
`experiments\10_YOLO26s_Binary8Class_PartialLabel_Baseline\run10_preregistered_config_v1.yaml`

- config SHA-256：
  `7480f3cc7fbfc6de35e769f2df4386cdadfb84b069ba2a2a9a0f277560f14c72`；
- complete behavior/data/code fingerprint：
  `a09e0cfb4669e06ff8ff67036699253282bbaa619c9bba28eb3e79bfb7445d9a`；
- YOLO26s-sem pretrained weights SHA-256：
  `bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435`。

关键参数：100 epochs、patience=20、batch=16、imgsz=640、AdamW、lr0=0.001、
cosine lrf=0.01、weight_decay=0.0005、warmup=3、seed=42、AMP=true、workers=8。
image classification loss weight=0.5；custom image-head dropout=0.20；框架 dropout=0；
AutoAugment/erasing/Mosaic/MixUp/CutMix/CopyPaste 全部显式关闭。

## 3. development-only 数据锁

实验本地派生 manifest 只投影 accepted combined manifest 的 train/val 行，不修改
任何 accepted source manifest、split、mask 或 test 规则：

- train：23,462；seg=10,483；cls=13,715；both=736；
- val：1,744；seg=292；cls=1,488；both=36；
- 634 个 IMA++ test 行不在 Run10 development manifest；data YAML 为 `test: null`；
- PAD-UFES-20 与 ISIC2018 external candidate 不在 YAML 或 development manifest。

派生 manifest SHA-256：
`3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`。  
派生 data YAML SHA-256：
`cc8d2b2063fdf2e60817651d2cc747c607aa2df5077eef5bc0f9a1608df750cc`。

用户手工解压的 PAD 三个目录已逐文件对 ZIP 的 size+CRC32 验收：911/659/728，
总计 2,298，PASS；其训练角色仍为 `none_external_candidate_only`。

## 4. 模型选择与断点锁

- val-only：`joint_score=sqrt(binary_iou * group_macro_f1)`；
- feasible：group mel->nv rate <= 0.1077844311 且 mel support 存在；
- primary alias：`best_feasible_joint`；
- tie：group Macro-F1、binary IoU、较早 epoch；
- 无 feasible checkpoint 时 Run10 判约束失败，不得借 test/external 晋升候选；
- semantic threshold=0.5、classification=argmax、temperature=1.0。

每个完整 epoch 发布 checkpoint bundle v2，包含 live FP32 model、EMA、optimizer、
AMP scaler、scheduler、early stopper、selection state、全部 RNG 和 data progress；
`resume=auto` 只接受 `_SUCCESS`、文件哈希与 fingerprint 全部匹配的 bundle。

连续/中断恢复 preflight 均 PASS：CPU、RTX 5080 AMP-off、RTX 5080 AMP-on +
workers=2，模型/EMA/optimizer/scaler/scheduler/stopper/selection/RNG/批次顺序比较均
无失败。

## 5. 点击前最终验收

`RUN10_PREFLIGHT_REPORT.json` 与 `RUN10_READY_TO_TRAIN.json` 均 PASS：

- RTX 5080，PyTorch 2.8.0+cu129，CUDA 12.9；
- 正式 batch=16、640、AMP 的真实 development forward+loss PASS；
- peak CUDA allocated=593,635,328 bytes，reserved=1,002,438,656 bytes；
- model state 未改变；没有创建 optimizer；没有 backward/optimizer step；
- `trainer.train()` 未调用；正式 results 目录不存在；
- 9 个 partial-label/checkpoint unittest 全部 PASS；
- 正式 overrides + Trainer constructor PASS，且 train 未调用。

## 6. 点击后边界

训练完成或正常 early stopping 后，返回 Codex 做 val-only checkpoint 锁定。仍禁止
自行运行 IMA++ test、ISIC2018 external 或 PAD external。只有模型、配置、checkpoint
和 evaluator SHA-256 全部冻结，并由用户再次明确授权后，才能分别执行 external 与
internal blind test；结果不得回流 Run10。
