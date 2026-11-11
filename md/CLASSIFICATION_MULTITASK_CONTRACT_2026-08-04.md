# PiFuV2 分类与 partial-label 多任务契约 v1

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`CLASSIFICATION_TAXONOMY_GATE=PASS`  
`CLASSIFICATION_DATA_ID_DEDUP_GATE=PASS`  
`CLASSIFICATION_DATA_FILE_DEDUP_GATE=BLOCKED_PENDING_IMAGE_AUDIT`  
`PARTIAL_LABEL_MODEL_GATE=BLOCKED_PENDING_IMPLEMENTATION_AND_SMOKE`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

本文件只锁定分类 taxonomy、标签来源、partial-label 语义和候选模型选择协议。它不授权创建或启动 `experiments\10_...`，也不改变 IMA++ 的既有 manifest、split、掩码值域或 test 规则。

## 1. 分类 taxonomy

新分类头固定为 8 类，类别 ID 保留历史七类顺序并在末尾新增 SCC：

| class_id | canonical_name | ISIC 2019 官方列 | 含义 |
|---:|---|---|---|
| 0 | akiec | AK | 光化性角化病/鲍恩病相关病变 |
| 1 | bcc | BCC | 基底细胞癌 |
| 2 | bkl | BKL | 良性角化性病变 |
| 3 | df | DF | 皮肤纤维瘤 |
| 4 | mel | MEL | 黑色素瘤 |
| 5 | nv | NV | 黑色素细胞痣 |
| 6 | vasc | VASC | 血管性病变 |
| 7 | scc | SCC | 鳞状细胞癌 |

ISIC 2019 的 `UNK` 是官方挑战 test 才出现的 outlier 类；官方训练真值中 `UNK=0`，因此不进入训练 taxonomy，也不把未知诊断强行映射到上述八类。

机器可读定义见根目录 `classification_taxonomy_v1.yaml`。模型、loader、评价器和 checkpoint fingerprint 必须验证类名与顺序完全一致。

## 2. 可接受的分类标签来源

v1 只接受官方 `ISIC_2019_Training_GroundTruth.csv` 中恰好一个正类的 one-hot 标签。当前 25,331 行全部通过 one-hot 完整性检查。

- 不从二值掩码推导疾病类别；
- 不把 IMA++ 的层级诊断文本自动压缩成八类；
- 不使用旧 V3 广播到病灶像素的七病种标签生成新真值；
- 不使用 ISIC 2019 官方 test 图像、test 真值或 leaderboard 结果做模型选择；
- IMA++ 图像若与官方 ISIC 2019 训练真值同 ID，可获得分类标签，但必须继承 IMA++ 已锁定的 train/val/test 角色，不能因分类数据加入而换 split。

## 3. 官方分类数据 ID 去重

官方 ISIC 2019 训练集共 25,331 张：MEL=4,522、NV=12,875、BCC=3,323、AK=867、BKL=2,624、DF=239、VASC=253、SCC=628。

训练候选必须先排除：

1. 官方 ISIC 2018 Task 3/HAM10000 lesion grouping 中的全部 10,015 个 ID；
2. IMA++ 原始 `img_metadata.csv` 中的全部 14,967 个 ID，而不只排除进入二值分割 v1 的 11,409 个 ID。

在 ISIC 2019 内，HAM10000 命中 10,015 个 ID，IMA++ 命中 828 个 ID，两组交集为 0；联合排除 10,843 个 ID，留下 14,488 个 ID 级候选：

| 类别 | 候选数 |
|---|---:|
| akiec | 737 |
| bcc | 2,809 |
| bkl | 1,523 |
| df | 124 |
| mel | 3,231 |
| nv | 5,522 |
| vasc | 111 |
| scc | 431 |

这 14,488 只是 ID 去重候选。在官方图像本体完成下载、ZIP 完整性验证、SHA-256 与严格 `pHash+dHash+尺寸` 审计之前，不得标记 `CLASSIFICATION_DATA_FILE_DEDUP_GATE=PASS`。

## 4. split 与 partial-label 规则

- IMA++ 11,409 个二值分割样本的既有 split 不变；其 828 个官方分类标签重叠中，train=736、val=36、test=56。
- IMA++ test 的 56 个分类标签只记录 provenance；正式模型确定和用户明确授权前不得加载或评价。
- ISIC 2019 增量候选只从官方 training 数据内部建立项目 train/val，不建立或读取官方 challenge test。
- 分类增量按 `lesion_id`、SHA-256 和严格感知重复签名绑定为不可拆分组；空 `lesion_id` 的样本只能作为 singleton，且必须在局限性中报告 patient ID 缺失。
- 同一组出现多个疾病类时构建失败，不允许随机拆分或多数票掩盖冲突。
- split 在看模型结果前由固定 seed 产生；生成后写入新的分类 manifest 并冻结。

每个 batch 样本必须显式包含：

- `has_segmentation`：是否具有真实 0/1 病灶 mask；
- `has_classification`：是否具有官方八类标签；
- `classification_target`：有标签时为 `0..7`，否则为 `-1`；
- `dataset_role`、`split`、`group_id` 和原始 `isic_id`。

分类-only 样本的语义目标使用全 `255` 的 ignore mask 作为 loader 占位，但 manifest 必须声明 `has_segmentation=0`。该占位不是病灶真值，不能进入分割指标或被写成新掩码标注。

## 5. 多任务模型与 loss 契约

- 分割头：`nc=1`，单通道 lesion logit，真实磁盘掩码仍为 `0=background, 1=lesion`；letterbox/无分割监督位置可为 `255=ignore`。
- 分类头：独立 8-logit 图像分类头；v1 干净基线不加入 CBAM、ECA、病灶引导注意力、lesion-balanced sampler 或 label smoothing。
- `L_seg` 只在 `has_segmentation=1` 的有效像素上计算；`L_cls` 只在 `has_classification=1` 的样本上计算。
- 某个 batch 没有某任务标签时，该任务 loss 必须为同设备、同 dtype 的可微零值，不能产生 NaN。
- v1 禁止 Mosaic、MixUp、CutMix 和 CopyPaste，因为它们会把多张图的分类标签混成一个未定义目标；只允许保持单图标签语义的几何/颜色增强。
- task loss 权重、采样比例、初始化权重、优化器、训练预算和随机种子必须在未来实验协议中另行预注册；本文件不创建 Run10 配置。

## 6. val 指标与模型选择候选协议

统一 val 至少输出：

- 二值病灶 IoU、Dice、pixel accuracy；
- 8 类 image-level 和 lesion/group-level Accuracy、Macro-Precision、Macro-Recall、Macro-F1；
- 每类 precision/recall/F1 与 support；
- `mel->nv` 数量和比例、`nv->mel` 数量和比例；
- malignant (`mel/bcc/scc`) 被判为 benign 的数量和比例；
- NLL、multi-class Brier 和 ECE；
- lesion/group 分层 bootstrap 95% CI；
- 参数量、GFLOPs 和真实设备推理速度。

候选 checkpoint 同时保存：

1. `best_binary_iou`；
2. `best_macro_f1`；
3. `best_joint_score`，其中 `joint_score=sqrt(binary_iou * lesion_macro_f1)`；
4. `best_feasible_joint`，要求 `mel_to_nv_rate <= 0.1077844311` 后再按 `joint_score` 排序。

若 val 没有满足风险约束的 checkpoint，实验结论必须是“无合规 checkpoint”，不能用 `best_joint_score` 冒充正式候选。该候选协议仍需在正式 Run10 协议中由用户确认后生效。

## 7. 完整断点恢复契约

新训练 checkpoint 除模型权重外必须保存并恢复：EMA、optimizer、scheduler、AMP scaler、已完成 epoch/global step、Python/NumPy/Torch CPU/CUDA RNG、sampler/data epoch、EarlyStopping 状态、四类专项 best 状态、配置与数据 manifest fingerprint、Ultralytics/PyTorch/CUDA 版本。

- 保存到临时目录或临时文件，完整写入并 fsync 后再原子发布；
- 只有带 `_SUCCESS` 且全部必需项通过验证的 checkpoint 才可被 `resume=auto` 选择；
- 显式 resume 与 auto resume 都必须拒绝 stripped、缺 optimizer 或 fingerprint 不匹配的 checkpoint；
- SIGINT/SIGTERM 只请求在安全 step/epoch 边界保存，不能在半个 optimizer step 中间伪造完整状态；
- 周期轮换只删除已验证完整且不再活跃的旧 checkpoint。

## 8. 当前门禁

在以下事项完成前，`PROJECT_RUN10_GATE` 保持阻塞：

- 官方图像 ZIP 完整性及 14,488 候选的文件级严格去重；
- 新分类 manifest/split 独立验收；
- partial-label loader、模型、validator 和完整 checkpoint contract 的 CPU/静态测试；
- RTX 5080 上真实 forward/backward、保存/恢复一致性冒烟；
- 真正独立外部域或最终锁定盲测角色；
- 最新 resolution 文档明确写出 `PROJECT_RUN10_GATE=PASS`；
- 用户明确确认可以设计并启动 `experiments\10_...`。

