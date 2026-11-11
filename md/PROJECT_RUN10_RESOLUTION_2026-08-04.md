# PiFuV2 Run10 前置实现与门禁决议

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`CLASSIFICATION_TAXONOMY_GATE=PASS`  
`CLASSIFICATION_DATA_ID_DEDUP_GATE=PASS`  
`CLASSIFICATION_DATA_FILE_DEDUP_GATE=PASS`  
`CLASSIFICATION_INCREMENTAL_DATASET_GATE=PASS`  
`PARTIAL_LABEL_LOADER_GATE=PASS`  
`MULTITASK_MODEL_IMPLEMENTATION_GATE=PASS_WITH_CPU_AND_SINGLE_BATCH_GPU_SMOKE`  
`CHECKPOINT_CONTRACT_UNIT_GATE=PASS`  
`UNIFIED_VAL_PROTOCOL_IMPLEMENTATION_GATE=PASS_STATIC_ONLY`  
`TEST_METRICS_USED=FALSE`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

本决议记录当前前置工作的真实完成度，不授权创建、设计或启动
`experiments\10_...`。只有后续最新文档明确写出
`PROJECT_RUN10_GATE=PASS`，并且用户再次明确确认后，才可进入 Run10。

## 1. 保留边界

- `experiments\01_...` 至 `experiments\09_...` 均保留；没有创建
  `experiments\10_...`。
- Run9 的既有事实仍为 epoch 69 正常 early stopping；本阶段没有重跑或改写。
- 旧 V3 仍仅用于历史复现。
- 没有改动 `Data\processed_imaplusplus_binary_v1` 的 manifest、split、0/1
  掩码值域或 test 规则。
- 没有改动 `Data\processed_isic2018_task1_incremental_v1`；该 89 对数据仍只作
  外部候选，不进入训练。
- 没有读取或评价 IMA++ test，也没有读取 ISIC 2019 challenge test。

本阶段记录的受保护 manifest SHA-256：

- IMA++ binary v1：
  `47C3763F210F36D9428A0D0B90427451EB6D5F43B43163BCB7141C191614396E`
- ISIC 2018 Task 1 incremental v1：
  `595FBBFC279AB5B591990E9BC266CDA337FDB3646F1BC277605E4F23B8075102`

## 2. 分类 taxonomy 与标签来源

机器可读 taxonomy 已锁定在 `classification_taxonomy_v1.yaml`：

| class_id | canonical_name | 官方 ISIC 2019 列 |
|---:|---|---|
| 0 | akiec | AK |
| 1 | bcc | BCC |
| 2 | bkl | BKL |
| 3 | df | DF |
| 4 | mel | MEL |
| 5 | nv | NV |
| 6 | vasc | VASC |
| 7 | scc | SCC |

只接受官方 `ISIC_2019_Training_GroundTruth.csv` 中恰好一个正类的 one-hot
标签。官方 training 的 25,331 行全部通过检查；`UNK` 在 training 中为 0，
不进入八类 taxonomy。不从 IMA++ 层级诊断、二值 mask 或旧 V3 像素类别反推
新分类真值。

## 3. 官方分类数据严格去重与冻结 split

数据目录：`Data\processed_isic2019_classification_incremental_v1`

- 官方 training：25,331；HAM10000 ID 排除 10,015；IMA++ 原始 ID 排除 828。
- ID 候选 14,488；跨参考池文件/严格感知重复排除 2；候选内部严格重复排除 55。
- 最终保留 14,431；train=12,979，val=1,452。
- 最终类别计数：akiec=729、bcc=2,801、bkl=1,509、df=124、mel=3,219、
  nv=5,507、vasc=111、scc=431。
- split 在训练前按 lesion ID、SHA-256、严格 `pHash+dHash+尺寸` 绑定组，固定
  seed=42；空 lesion ID 共 1,256 个，作为 singleton 并保留局限性记录。
- 分类-only mask 全为 255，只是 loader 的 ignore 占位；manifest 明确
  `has_segmentation=0`，不得解释为病灶真值。
- 数据目录包含 `_SUCCESS`、manifest、group manifest、排除清单、重复组、
  SHA-256 清单、provenance 和构建报告。

独立只读验收已完整解码 14,431 张图像及占位 mask，确认尺寸一致、mask 全 255、
split 数量一致、没有缺失/损坏文件；另抽样重新计算 128 对共 256 个文件哈希。
验收没有调用模型，也没有产生 test 指标。

## 4. partial-label loader 与多任务模型

已实现：

- `tools\imaplusplus_partial_multitask_v1.py`
- `tools\verify_partial_data_loader_v1.py`

联合 manifest 明确提供 `has_segmentation`、`has_classification`、
`classification_target`、`dataset_role`、`split`、`group_id` 和 `isic_id`。
真实联合 loader 验收结果：

| split | 总数 | 分割监督 | 分类监督 | 双监督 |
|---|---:|---:|---:|---:|
| train | 23,462 | 10,483 | 13,715 | 736 |
| val | 1,744 | 292 | 1,488 | 36 |

cache 只写入 `Data\.cache\partial_multitask_v1`，不写进受保护的新数据目录。
Mosaic、MixUp、CutMix、CopyPaste 均要求关闭。

模型基线为 YOLO26s 二值语义头加独立 8-logit 图像分类头；没有加入注意力、
特殊 sampler 或 label smoothing。分割 loss 只作用于有分割真值的样本，分类
loss 只作用于有官方分类标签的样本；无对应任务标签的 batch 返回可微零值。

## 5. 断点恢复实现边界

`tools\multitask_checkpoint_contract.py` 已实现完整 bundle 契约：模型/EMA、
optimizer、scheduler、AMP scaler、epoch/global step、EarlyStopping、模型选择状态、
Python/NumPy/Torch CPU/CUDA RNG、sampler/data epoch、loader generator、配置/数据
fingerprint 和运行时版本。只有必需文件哈希通过且含 `_SUCCESS` 的 bundle 才能被
`resume=auto` 选择；显式恢复也检查 fingerprint。SIGINT/SIGTERM 采用安全边界
请求，不在半个 optimizer step 中发布完整 checkpoint。

单元测试已覆盖：完整保存、auto/latest、拒绝不完整目录、fingerprint 不匹配、
轮换并保留专项 best。当前尚未在完整 Ultralytics 多 epoch 训练器中执行一次真实
“训练若干 step -> 中断 -> auto resume -> 状态与轨迹一致”冒烟，因此这一项仍是
Run10 前的真实阻断项，不能仅凭单元测试宣称端到端恢复已验收。

## 6. 统一 val 指标协议

`tools\evaluate_imaplusplus_multitask_v1.py` 已实现以下 val 输出协议：

- binary lesion IoU、Dice、pixel accuracy；
- image-level 与 lesion/group-level accuracy、macro precision/recall/F1；
- 八类逐类 precision/recall/F1/support；
- mel->nv、nv->mel、malignant(mel/bcc/scc)->benign 风险；
- NLL、multiclass Brier、ECE；
- group bootstrap 95% CI；
- `joint_score=sqrt(binary_iou * group_macro_f1)`；
- `mel_to_nv_rate <= 0.1077844311` 的 feasible 约束。

评价入口默认且当前只允许 val。test 必须同时显式指定 `--split test` 和
`--allow-test`，并且仍需满足正式模型已锁定与用户授权。因为当前没有 Run10 训练
checkpoint，本阶段只完成协议实现与合成测试，没有报告任何真实 val 模型指标。

## 7. 当前验证等级

- 项目环境：Python 3.13.7、PyTorch 2.8.0+cu129；
  `torch.cuda.is_available()=True`，设备为 NVIDIA GeForce RTX 5080。
- 新增 7 个工具脚本通过 `py_compile`。
- 8 个 `unittest` 全部通过。
- 合成 CPU 模型 smoke 通过。
- RTX 5080 上实际联合 train batch（classification-only、joint、
  segmentation-only 各 1 个）完成 forward、partial loss、backward 和 optimizer
  step；两个任务头均得到非零梯度。
- 上述 GPU 结果是单批实现冒烟，不是训练结果，不构成模型优劣或 Run10 放行证据。

## 8. 保持阻塞的事项

`PROJECT_RUN10_GATE` 继续保持
`BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`，原因是至少还缺：

1. 完整训练器中的周期保存、SIGINT/SIGTERM、`resume=auto` 和恢复一致性 GPU 冒烟；
2. 真正独立外部域或最终锁定盲测角色及其与 IMA++ test 的边界协议；
3. Run10 的单一预注册因素、初始化权重、loss 权重、采样比例、优化器、预算、seed、
   early-stopping 和候选选择规则；
4. 一份更新的最终 resolution 明确写出 `PROJECT_RUN10_GATE=PASS`；
5. 用户随后明确确认可设计并启动 `experiments\10_...`。

在这些条件全部满足前，不创建 Run10 目录、不启动正式训练、不运行 test。

