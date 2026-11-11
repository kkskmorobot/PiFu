# PiFuV2 新语义分割数据处理与验收报告

更新时间：2026-08-04  
实际项目根目录：`E:\App\DaiMa\JetBrains\Project\PiFuV2`  
原始下载目录：`Data\1`  
本轮行为：只做数据整理、去重、划分、质量审计和 loader 验证；没有启动训练，也没有计算任何模型 test 指标。

## 1. 状态结论

`BINARY_SEGMENTATION_DATA_GATE=PASS`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

IMA++ 已处理为可以被当前 Ultralytics `SemanticDataset` 直接读取的二值语义分割数据。项目总目标若仍包含疾病分类，则还需要锁定分类 taxonomy、分类数据、partial-label loader、多任务模型结构和新的模型选择约束，不能仅凭本报告直接启动旧八通道 Run10。

## 2. IMA++ 主数据

输出目录：`Data\processed_imaplusplus_binary_v1`

### 2.1 标签定义

- 任务：病灶/背景二值语义分割。
- 输出 PNG 掩码：`0=background`、`1=lesion`。
- Ultralytics letterbox 时可临时产生 `255=ignore padding`，这是当前 loss 的合法输入，不存在于磁盘原始输出掩码中。
- 多标注图像：使用 IMA++ STAPLE 共识掩码。
- 单标注图像：首个干净基线只使用专家级 `S1`。
- 原始诊断层级只保存在 `manifest.csv`，没有广播到病灶像素，也没有强行映射为旧七类像素标签。

### 2.2 数量

| 项目 | 数量 |
|---|---:|
| IMA++ 原图 | 14,967 |
| 原始掩码 | 22,472 |
| 进入处理后数据 | 11,409 |
| train | 10,483 |
| val | 292 |
| test | 634 |
| STAPLE 共识掩码 | 2,392 |
| S1 单标注掩码 | 9,017 |
| 排除总数 | 3,558 |
| 其中 S2 非专家单标注 | 3,556 |
| 其中空掩码 | 1 |
| 其中全图掩码 | 1 |

两个客观无效的共识掩码已隔离，不会进入任何 split。已确认的全图异常之一为 `ISIC_0010065`。

### 2.3 划分与泄漏控制

IMA++ 官方 1,675/240/479 多标注划分被用作锚点，而不是被盲目照搬。全部有效样本按以下字段绑定为不可拆分组：

- `patient_id`；
- `lesion_id`；
- 原图 SHA-256；
- 严格相同的 pHash+dHash+图像尺寸签名。

若官方锚点冲突，按 `test > val > train` 把整个关联组移向更严格的保留集合。本轮共发现 71 个官方锚点冲突组。典型实例中，`ISIC_0000442` 与 `ISIC_0023783` 是同一图像的不同编码副本，而 `ISIC_0023784` 是同一患者的另一视图；三者已整体放入 test，避免 train/test 泄漏。

验收结果：

- train/val/test 的非空 `patient_id` 两两交集为 0；
- train/val/test 的非空 `lesion_id` 两两交集为 0；
- 10,595 个自动关联组均只属于一个 split；
- 未发现输出内完全相同 SHA-256 的重复组；
- 310 个严格感知重复组已绑定在同一 split。

### 2.4 质量标记

136 个有效样本进入 `quality_flags.csv`，其中：

- 97 个掩码连通域多于 10；
- 37 个病灶面积高于图像面积 95%；
- 3 个病灶面积低于图像面积 0.1%。

这些不是自动判死的错误：抽查显示其中既有碎片化共识边界、近全幅病灶，也有真实的极小病灶。因此 v1 保留它们并显式标记，不擅自修改官方边界。后续可以预注册一次“保留全部 vs 质量标记过滤/最大连通域”的敏感性实验，但不得看过结果后临时挑规则。

### 2.5 可复现文件

- `data.yaml`：当前 Ultralytics 二值语义数据入口；
- `manifest.csv`：每个有效样本的来源、split、group、mask 策略、许可、诊断原字段、尺寸和哈希；
- `excluded.csv`：全部排除样本与原因；
- `quality_flags.csv`：自动质量标记；
- `group_manifest.csv`：防泄漏关联组与官方锚点冲突处理；
- `duplicate_groups.csv`：SHA/感知重复组；
- `hashes_sha256.csv`：处理后文件哈希；
- `provenance\`：官方元数据、split 清单与许可说明；
- `_SUCCESS`：只有完整构建成功后才存在。

## 3. ISIC 2018 Task 1

输出目录：`Data\processed_isic2018_task1_incremental_v1`

### 3.1 完整性与交集

- 原图：2,594；
- Task 1 lesion/background GroundTruth：2,594；
- 原图与掩码缺失配对：0；
- 与 IMA++ 同 ISIC ID：2,505；
- ID、SHA-256、严格感知哈希去重后的增量：89。

因此完整 ISIC 2018 Task 1 不能作为 IMA++ 训练后的独立外部集。89 张增量已单独保存，角色固定为 `EXTERNAL_CANDIDATE_NOT_FOR_TRAINING`。它仍来自 ISIC 域，且当前缺少 patient/lesion 元数据，不能宣称是独立医院或独立临床盲测。

入口文件为 `external_eval.yaml`，没有提供 train split，防止误用为训练增量。

### 3.2 ISIC 2018 Task 2

`Data\1\ISIC2018_Task2_Training_GroundTruth_v3` 中有 12,970 张属性掩码，即 5 类皮肤镜属性乘以 2,594 张图像。它们不是 Task 1 的病灶/背景主掩码，本轮保持原样，没有混入二值分割真值。

## 4. 独立验收结果

在构建脚本之外重新执行了以下检查：

- 全量读取 IMA++ 11,409 对和 ISIC 2018 增量 89 对；
- 图像/掩码一一配对；
- 尺寸一致；
- 磁盘掩码值域严格为 `{0,1}`；
- 掩码非空且非全图；
- 两个数据集各抽样 128 个输出文件重新计算 SHA-256，全部匹配；
- patient/lesion/group 跨 split 泄漏为 0；
- IMA++ 与 ISIC 2018 源文件数量保持完整；
- 当前项目 Ultralytics loader 分别读取 train、val、test、external 样本成功，张量形状为 `image=(3,640,640)`、`mask=(640,640)`，值域为 `{0,1,255}`。

结论：`FINAL_INDEPENDENT_VALIDATION=PASS`，其中 loader 的 255 仅为 letterbox ignore padding。

## 5. 源数据保护

- `Data\1` 中的下载文件未被删除、移动、重命名或原地修改；
- 旧 `Data\processed_skin7_v3` 和 experiments 01~09 未修改；
- 新脚本支持 IMA++ 临时 `.building` 目录与 `_SUCCESS` 原子发布；
- 新数据与旧 V3 是并列版本，不是覆盖关系。

## 6. 下一步门槛

如果下一步只建立纯二值语义分割新基线，可以使用 `Data\processed_imaplusplus_binary_v1\data.yaml` 设计新模型配置，但仍需用户明确确认后才能训练。

如果下一步继续项目原定的“分割 + 疾病分类”，还必须完成：

1. 锁定新的疾病分类 taxonomy 与映射；
2. 决定并获取官方分类数据，去除与 IMA++ 的 patient/lesion/ID/哈希重复；
3. 实现有掩码/仅分类标签均可训练的 partial-label loader；
4. 将旧八通道疾病语义头改为二值分割头 + 图像分类头；
5. 重新预注册 binary Dice/IoU、分类 Macro-F1、风险约束和模型选择规则；
6. 确定真正独立的外部域或锁定盲测集；
7. 用户确认后再启动 experiments 10，不沿用旧八通道 checkpoint 直接续训。

正式模型选择前仍禁止读取模型 test 指标。
