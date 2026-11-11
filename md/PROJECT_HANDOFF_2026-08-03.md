# PiFuV2 项目交接文档（训练工作流）

更新时间：2026-08-04  
项目根目录：`E:\App\DaiMa\JetBrains\Project\PiFuV2`  
本文档用途：把后续正式训练、监控、断点恢复和结果验证交给下一位 AI，并记录 2026-08-04 已完成的 IMA++/ISIC 2018 数据治理结果及仍未通过的项目级启动门槛。  
状态口径：本文只记录已经从磁盘、日志或项目代码核实的事实。尚未完成的事项不会为了方便交接而写成“已经解决”。

---

## 0. 新 AI 必须先读的最短结论

1. 这是一个基于皮肤镜图像的医学图像分析研究项目，目标是同时完成病灶像素级分割和七类皮肤病识别，并研究注意力、长尾学习和风险敏感模型选择。
2. 项目已经完成 9 次正式实验。最新 Run9 在 epoch 69 正常早停，不是崩溃；最佳 Macro-F1 位于 epoch 49，唯一满足预注册三项约束的 checkpoint 位于 epoch 37。
3. 当前没有 PiFuV2 Python 训练进程。Run9 的关键权重、日志、曲线和周期 checkpoint 已完整落盘。
4. 历史正式数据 `Data\processed_skin7_v3` 仍完整保留，train/val/test=`8512/1504/1503`；不得删除、重建或覆盖。它只服务于 Run1~Run9 复现和历史对照。
5. 新的 IMA++ 二值语义数据已经完成构建和独立验收：`Data\processed_imaplusplus_binary_v1`，train/val/test=`10483/292/634`，磁盘掩码为 `0=background, 1=lesion`，`BINARY_SEGMENTATION_DATA_GATE=PASS`。
6. ISIC 2018 Task 1 的 2,594 对原图/掩码已经审计，其中 2,505 个 ID 与 IMA++ 重叠；严格去重后仅 89 对进入 `Data\processed_isic2018_task1_incremental_v1`，角色固定为外部候选、不得用于训练，也不能宣称为独立医院外部域。
7. 历史模型仍是八通道“七病种+背景”，但这种标签由图像级诊断广播到病灶区域。新数据已明确改为二值病灶语义；若保留疾病识别，推荐实现独立图像分类头，不再制造七病种像素真值。
8. **现在仍禁止启动 Run10。** 分割数据已通过，但分类 taxonomy、官方分类数据去重、partial-label loader、多任务模型实现、新模型选择约束和真正外部盲测仍未锁定，故 `PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`。
9. 最终模型确定前禁止读取或运行 test。统一评价默认只能用 val；test 需要用户明确授权，并同时传入 `--split test --allow-test`。
10. 不得覆盖 `experiments\01_...` 到 `experiments\09_...`。未来正式实验从 `experiments\10_...` 开始。
11. 从备份恢复只能复制，不能剪切、移动或删除备份原件。
12. 本文件比根目录中仍写着“Run9 训练中”的旧 `PROJECT_INDEX.md` 更新；数据事实以 `DATA_TASK_ACCEPTANCE_2026-08-04.md`、本文件和实际磁盘为准。

---

## 1. 工作交接边界

### 1.1 下一位训练 AI 的职责

- 在前置数据/任务验收通过后，设计严格控制变量的 Run10 配置；
- 完成 dry-run、CPU/静态检查、RTX 5080 小规模冒烟和正式 PyCharm 启动；
- 为正式训练保留完整断点恢复机制；
- 监控训练是否正常、判断早停是否按预注册指标触发；
- 只使用 val 完成统一评价、混淆矩阵、病灶级 bootstrap 和效率评估；
- 记录实验唯一变量、配置、权重来源、随机种子和结果；
- 不自行修改数据划分、标签语义、类别顺序或 test 使用规则。

### 1.2 用户与当前 AI 的数据/任务工作状态

已经完成：

- IMA++ 官方原图、掩码、标注技能、许可和元数据验真；
- IMA++ 单一训练掩码策略：多人标注使用 STAPLE，单人标注 v1 只纳入 S1；
- 空/全图掩码隔离、极端面积/多连通域自动标记；
- patient、lesion、SHA-256 和严格感知哈希防泄漏分组；
- ISIC 2018 Task 1 与 IMA++ 的 ID、文件哈希和感知重复审计；
- 二值 PNG-mask 数据契约、机器可读 manifest、逐文件哈希和 Ultralytics loader 验收；
- 数据验收报告 `DATA_TASK_ACCEPTANCE_2026-08-04.md`。

仍需完成：

- 锁定疾病分类 taxonomy、类别数和诊断映射；
- 获取/审计官方分类数据并去除与 IMA++、HAM10000 的重复；
- 实现“二值分割头 + 图像分类头”及 partial-label loader；
- 重新预注册模型选择指标、风险约束和训练预算；
- 确认真正独立的外部域或最终锁定盲测集；
- 用户明确确认可以启动 Run10。

### 1.3 如何理解“上述问题已经解决”

截至 2026-08-04，二值分割数据治理已经完成并有独立验收报告，但整个多任务项目门槛仍未通过。下一位训练 AI 不需要重复调查 IMA++/ISIC 2018 的文件配对、掩码值域或当前 split；它必须继续等待最新 resolution 文档明确写出 `PROJECT_RUN10_GATE=PASS` 且用户确认，才能开始 Run10。

---

## 2. 整体任务目标

### 2.1 工程目标

构建一个可复现、可恢复、可审计的皮肤镜图像模型，要求：

- 定位病灶并输出像素级分割区域；
- 识别七类皮肤病；
- 在严重类别不平衡下不过度偏向 `nv`；
- 提升 `akiec`、`bcc`、`df` 等少数类表现；
- 重点控制 `mel→nv`，避免黑色素瘤被误判为普通痣；
- 保存完整训练状态和周期 checkpoint，支持中断后继续；
- 每次正式实验只改变一个预注册因素；
- 保留所有稳定基线、训练日志和正式结果，不覆盖历史实验。

### 2.2 论文/研究目标

项目不是单纯追求 Accuracy，而是要回答：

1. HAM10000 七类病灶识别与分割的主要瓶颈是什么？
2. 长尾类别、重复病灶、离线增强、设备/来源域偏差如何影响模型？
3. 注意力模块是否真正让模型关注病灶颜色、纹理和边界？
4. 改进是否在控制数据、loss、优化器、训练轮数和随机种子后仍成立？
5. 分类性能、分割性能和临床风险方向能否同时改善？
6. 模型在外部数据、不同医院/设备和不同肤色上是否仍可靠？

### 2.3 最终成功标准

最终模型至少应同时报告：

- 前景分割 mIoU；
- 图像级 Accuracy、Macro-Precision、Macro-Recall、Macro-F1；
- 少数类召回率；
- `mel→nv` 和 `nv→mel` 数量/比例；
- 病灶级而非只按图片级的指标和 bootstrap 95% CI；
- 参数量、GFLOPs、推理速度；
- 多随机种子或 lesion 级交叉验证；
- 概率校准指标；
- 外部验证结果；
- 方法完全锁定后的一次最终 test 结果。

---

## 3. 项目类型及其原因

### 3.1 当前技术类型

- 医学图像分析；
- 皮肤镜图像七分类；
- 像素级语义/实例分割；
- 长尾类别学习；
- 多任务学习；
- 基于注意力机制的 YOLO 改进研究；
- 风险敏感模型选择与外部泛化研究。

### 3.2 为什么项目需要语义分割

项目不仅要回答“是什么病”，还要回答“病灶在哪里”。像素级监督可以用于病灶边界、空间注意力、mask-guided pooling、可解释可视化和定位质量分析。因此不能简单退化成普通图片分类项目。

### 3.3 当前语义标签契约

当前 YOLO26 semantic 输出固定为：

| ID | 类别 | 含义 |
|---:|---|---|
| 0 | akiec | 光化性角化病/鲍恩病相关病变 |
| 1 | bcc | 基底细胞癌 |
| 2 | bkl | 良性角化性病变 |
| 3 | df | 皮肤纤维瘤 |
| 4 | mel | 黑色素瘤 |
| 5 | nv | 黑色素细胞痣 |
| 6 | vasc | 血管性病变 |
| 7 | background | 背景像素 |

独立语义评价必须使用：

`experiments\04_YOLO26s_Semantic_C2PSA\data_semantic_eval_v4.yaml`

其中必须保持：

- `nc: 8`；
- `bg_class_idx: 7`；
- 完整的 `0~7` 名称映射。

### 3.4 当前任务定义的关键方法学问题

当前数据并不是病理学意义上的逐像素七病种标注。实际制作过程是：

1. 每张图有一个病灶/背景二值掩码；
2. 每张图有一个图像级疾病诊断；
3. 将整个病灶区域统一填成该图像的疾病类别。

因此当前八通道模型把“病灶定位”和“图像诊断”压在同一像素分类头上。这很可能造成 Run5~Run9 反复出现的负迁移：分类提高时，前景 mIoU 或风险约束下降。

目前推荐优先评估的重构方案是：

- 二值病灶语义分割头：lesion/background；
- 七类或八类图像分类头：疾病诊断；
- 分割概率用于引导分类特征；
- 有掩码的样本计算分割 loss；
- 只有疾病标签的样本只计算分类 loss。

该方案仍然属于语义分割项目，并能使用更多官方分类数据；但在正式锁定前不得开始训练。

2026-08-04 更新：新数据的分割标签已经锁定为 `lesion/background` 二值语义，并通过 loader 验收；尚未锁定的是分类 taxonomy、分类数据、双头模型和 partial-label loss，因此不能把已完成的数据标签决策误写成整个多任务方案已完成。

### 3.5 为什么它不是临床诊断产品

项目当前只在单一公开数据域上开发，尚未完成医院外部验证、不同设备验证、肤色公平性、概率校准和临床风险评估。论文中只能称为辅助识别算法研究，不得宣称可以直接用于临床诊断。

---

## 4. 环境、硬件与工程路径

- 操作系统：Windows；
- IDE：PyCharm；
- 项目根目录：`E:\App\DaiMa\JetBrains\Project\PiFuV2`；
- 项目解释器：`E:\App\DaiMa\JetBrains\Project\PiFuV2\.venv\Scripts\python.exe`；
- Python：3.13.7；
- PyTorch：2.8.0+cu129；
- Ultralytics：8.4.101；
- GPU：NVIDIA GeForce RTX 5080，16 GB；
- Run9 正式配置：imgsz=640、batch=16、workers=8、AMP=true；
- Run9 模型：6,616,038 参数，约 17.4 GFLOPs（融合前）；
- 正式训练建议继续从 PyCharm 可见启动，便于用户观察和控制。

根目录的 `yolo26n.pt` 是历史冒烟过程中下载的轻量权重，不是 Run9 正式模型，也不是下一轮默认初始化权重。正式模型预训练权重位于 `models`，正式实验权重位于各自 `experiments` 目录。

---

## 5. 当前磁盘结构

```text
PiFuV2/
├─ .venv/                          当前有效环境
├─ Data/
│  ├─ 1/                           IMA++、ISIC 2018 解压后的官方源文件；只读保留
│  ├─ processed_skin7_v3/          Run1~Run9 历史数据；不得覆盖
│  ├─ processed_imaplusplus_binary_v1/
│  │                                11,409 对二值语义主数据；_SUCCESS 已存在
│  └─ processed_isic2018_task1_incremental_v1/
│                                   89 对外部候选；不得用于训练
├─ docs/
│  ├─ PROJECT_HANDOFF_2026-07-20.md
│  └─ 其他历史文档
├─ experiments/
│  ├─ 01_YOLO11s_V1_Baseline/
│  ├─ 02_YOLO11s_V2_DataBalance/
│  ├─ 03_YOLO11s_V3_NV3200/
│  ├─ 04_YOLO26s_Semantic_C2PSA/
│  ├─ 05_YOLO26s_Semantic_AuxClassifier/
│  ├─ 06_YOLO26s_Semantic_AuxClassifier_CBAM/
│  ├─ 07_YOLO26s_Semantic_LesionGuidedDualAttention/
│  ├─ 08_YOLO26s_Semantic_LesionBalancedSampler/
│  └─ 09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing/
├─ models/                          项目预训练模型
├─ tools/                           训练、恢复、评价和数据脚本
│  ├─ prepare_imaplusplus_binary_v1.py
│  └─ prepare_isic2018_task1_incremental_v1.py
├─ PROJECT_INDEX.md                 目前对 Run9 状态已过期
├─ DATA_TASK_ACCEPTANCE_2026-08-04.md
│                                   新数据完整验收与剩余门槛
└─ PROJECT_HANDOFF_2026-08-03.md    本文档；内容已更新至 2026-08-04
```

非数据、非虚拟环境备份：

`E:\App\DaiMa\JetBrains\Project\BeiFen\PiFuV2`

备份历史审计：268 个文件、2,310,555,920 字节，逐文件 SHA-256 检查曾报告 0 missing、0 mismatch。恢复时只能 Copy，不得 Move。

---

## 6. 当前 V3 数据状态

### 6.1 图片级分布

| split | akiec | bcc | bkl | df | mel | nv | vasc | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 800 | 1000 | 1400 | 320 | 1400 | 3200 | 392 | 8512 |
| val | 49 | 77 | 165 | 18 | 167 | 1006 | 22 | 1504 |
| test | 49 | 77 | 165 | 17 | 167 | 1006 | 22 | 1503 |

### 6.2 真实原图、病灶和离线增强

训练集表面有 8512 张，实际上是：

- 5515 张独立来源原图；
- 4042 个独立 `lesion_id`；
- 2997 张离线增强副本，占训练图片约 35.2%；
- train/val/test 的 lesion_id 和 source_image_id 两两交集均为 0；
- val 和 test 没有离线增强图片。

训练集各类真实来源/独立病灶：

| 类别 | 真实来源图 | 独立病灶 | 离线增强图 |
|---|---:|---:|---:|
| akiec | 229 | 160 | 571 |
| bcc | 360 | 229 | 640 |
| bkl | 769 | 510 | 631 |
| df | 80 | 51 | 240 |
| mel | 779 | 431 | 621 |
| nv | 3200 | 2593 | 0 |
| vasc | 98 | 68 | 294 |

val 的独立病灶数尤其小：akiec=34、bcc=49、bkl=108、df=11、mel=91、nv=810、vasc=15。少量样本变化会明显影响 Macro-F1 和少数类 Recall。

### 6.3 V3 已验证的优点

- 按 `lesion_id` 划分，同一病灶的多张照片不跨 split；
- val/test 不做离线增强；
- 当前元数据、图片和标签数量一致；
- 多边形标签基础结构检查已通过；
- 类别顺序固定；
- Run8/Run9 的 lesion-balanced sampler 已避免同一病灶因为拥有更多图片而被额外偏爱。

### 6.4 V3 尚未解决的风险

- 极少数类的独立病灶仍太少；
- 35.2% 训练图片是离线重复变体，不增加病例或域多样性；
- 原始 full train 中大量真实 NV 曾因 cap 被排除；
- 诊断确认方式和采集来源与类别高度相关；
- bkl 内部异质性大；
- patient ID 不完整，lesion-safe 不等同于 patient-safe；
- 当前项目内已经删除原始 `Data\train`，重建必须使用用户另行保存的官方原始备份；
- 现有掩码是否全部来自权威人工标注，尚未建立逐样本 provenance；
- 历史上 V3 的建立参考过 V2 的 test 表现，因此现有 test 不再是严格意义上的从未触碰终极盲测集。

---

## 7. 已经解决、训练 AI 不需要重复处理的问题

以下工程问题已经解决并通过实际实验使用：

1. **同病灶跨 split 泄漏**：使用 lesion_id 划分，当前三个 split 的 lesion/source 交集为 0。
2. **YOLO26 background 映射错误**：standalone semantic validation 已固定为 `nc=8`、`bg_class_idx=7`。
3. **语义数据没有实例式 `cls/batch_idx`**：多任务分类 target 已从 semantic mask 的唯一前景类别恢复，并拒绝空/多前景类别异常。
4. **训练/验证 tuple 形状不一致**：多任务训练、验证和预测代码已分别处理。
5. **统一评价缺失**：`tools\evaluate_skin7_unified.py` 已能输出语义、图像级、lesion 级、bootstrap、混淆矩阵和效率指标。
6. **test 被无意调用**：统一评价器默认 val；test 需要双重显式参数。
7. **重复病灶曝光偏置**：Run8/Run9 使用固定类别配额，并在类内先均匀选择 lesion、再均匀选择该 lesion 的图像。
8. **Run9 分类头过度自信的单因素测试**：label smoothing=0.05 已实现、静态验证、GPU 冒烟并完成正式训练。
9. **中断恢复入口**：Run4~Run9 都有各自 resume 脚本；Run9 的 dry-run 会在不反序列化 pickle payload 的情况下审计 ZIP/结构。
10. **历史实验保存**：01~09 的正式目录均保留，不需要从散乱 `runs*` 目录猜测结果。

---

## 8. 九轮正式实验谱系

| Run | 模型/唯一变化 | 结束状态 | 主要结论 |
|---:|---|---|---|
| 01 | YOLO11s-seg + V1 | 已完成 | 初始稳定实例分割基线；Mask mAP50-95 日志峰值约 0.5801 |
| 02 | 更强数据平衡与 hard-example 增强 | epoch 58 早停 | Mask mAP50-95 日志峰值约 0.5306，明显下降 |
| 03 | 回到稳定增强，仅 NV cap=3200 | 已完成 | 稳定 YOLO11s 实例分割基线；val Mask mAP50-95≈0.5752，图像 Macro-F1≈0.6994 |
| 04 | YOLO26s semantic + 原生 C2PSA | epoch 91 早停 | 当前最强前景分割基线，foreground mIoU=0.538748 |
| 05 | Run4 + P3/P4 全局池化辅助七分类头 | epoch 84 停止 | 最佳选择状态分割降至 0.470054；普通辅助头没有形成全面提升 |
| 06 | Run5 + 标准 CBAM | epoch 57 早停 | 最佳 epoch 37；少数类 Recall 提升，但分割和 nv→mel 未解决，无合规 checkpoint |
| 07 | Run5 分类聚合改为病灶引导 P3/P4/P5 双注意力 | epoch 55 早停 | 无 checkpoint 满足全部约束；自定义注意力没有形成稳定全面优势 |
| 08 | 固定 Run7，只改为 lesion-balanced sampler | epoch 82 早停 | 首次得到合规 checkpoint；采样策略有效，但仍存在分类/分割冲突和分类过拟合 |
| 09 | 固定 Run8，只给图像分类 CE 加 label_smoothing=0.05 | epoch 69 早停 | 分类 Macro-F1 峰值提高，但峰值不满足分割及 mel→nv 约束；唯一合规状态的 Macro-F1 反而下降 |

注意：Run1~Run3 的 Mask mAP 与 Run4~Run9 的 semantic mIoU 不是同一指标，不能直接横向比较。跨任务比较应使用统一图像级诊断指标。

---

## 9. Run4~Run9 关键 val 结果

### 9.1 正式统一验证结果

| 模型/checkpoint | fg mIoU | image Acc | image Macro-F1 | minority Recall | mel→nv | nv→mel | lesion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run4 `best.pt` | **0.538748** | 0.809176 | 0.704120 | 0.533498 | 18 | 84 | 0.724283 |
| Run5 selected `best.pt` | 0.470054 | 0.751330 | 0.672016 | 0.631691 | 10 | 63 | 0.670156 |
| Run6 epoch 37 | 0.489603 | 0.805186 | 0.701557 | **0.629801** | 14 | 121 | 0.715276 |
| Run7 composite epoch 35 | 0.473481 | 0.793218 | 0.683146 | 0.607572 | 18 | 126 | 0.724177 |
| Run7 macro epoch 43 | 0.491179 | 0.815160 | 0.693490 | 0.519824 | 19 | 85 | 0.730219 |
| Run8 feasible epoch 60 | 0.519161 | 0.811835 | 0.704217 | 0.546417 | 18 | 102 | 0.741995 |
| Run8 macro epoch 79 | 0.522081 | 0.813830 | **0.712415** | 0.579812 | 25 | 89 | **0.743597** |

Run8 epoch 60 是当时唯一同时满足三项预注册约束的 checkpoint。Run8 epoch 79 的分类指标更高，但 `mel→nv=25`，超过允许的 18。

### 9.2 固定验收门槛

- foreground mIoU ≥ 0.518748；
- minority Recall ≥ 0.533498；
- mel→nv rate ≤ 0.1077844311，即 val 167 张 mel 中最多 18 张被判为 nv；
- 合规 checkpoint 按 val Macro-F1 排序；
- 不读取 test。

这些门槛来自 Run4 基线，除非用户和数据/任务验收报告明确重新定义，否则训练 AI 不得在训练途中修改。

---

## 10. Run9 最终状态与详细结论

### 10.1 受控实验定义

Run9 固定 Run8 的：

- Run7 病灶引导模型；
- lesion-balanced sampler；
- 每轮固定类别配额 `800/1000/1400/320/1400/3200/392`；
- 8512 draws/epoch；
- 语义主/辅助 loss；
- 图像分类 loss 权重 0.5；
- 初始化权重；
- AdamW、LR 和 cosine schedule；
- 增强、imgsz=640、batch=16；
- seed=42、deterministic=true；
- patience=20；
- val split 和 checkpoint 选择策略。

唯一变化：七类图像分类交叉熵使用 `label_smoothing=0.05`。

### 10.2 早停是否正常

正常。Run9 日志包含 epoch 1~69，共 69 行。最佳 Macro-F1 出现在 epoch 49，之后连续 20 个 epoch 没有刷新，因此在 epoch 69 触发 patience=20 早停。当前没有对应 Python 训练进程。

### 10.3 Run9 训练日志关键点

| checkpoint | image Macro-F1 | fg mIoU | minority Recall | mel→nv rate | 约合 mel→nv |
|---|---:|---:|---:|---:|---:|
| epoch 37，best feasible | 0.68511 | 0.52111 | 0.61252 | 0.10778 | 18/167 |
| epoch 49，best macro | **0.72935** | 0.50159 | 0.58534 | 0.11976 | 20/167 |

Run9 的分类验证 loss 在 epoch 37 最低：

- train image CE=0.37855；
- val image CE=0.71891。

到 epoch 69：

- train image CE=0.26955；
- val image CE=0.78961。

由于使用 label smoothing，Run8 与 Run9 的 CE 绝对值不能直接当成同一尺度比较，但 Run9 仍然存在训练/验证间隙和中后期收益不稳定。

### 10.4 Run9 的正确解读

- label smoothing 提高了分类 Macro-F1 峰值；
- 但 epoch 49 的 fg mIoU 和 mel→nv 不合规；
- 唯一合规的 epoch 37，Macro-F1 低于 Run4/Run8 合规状态；
- Run4 仍是前景分割最强基线；
- Run9 没有解决分类、分割和风险约束不能同时最优的问题；
- Run9 目前尚未由 standalone unified evaluator 对 epoch 37 和 epoch 49 完成最终统一 val 复核，因此训练日志数字不能被写成最终论文数字。

### 10.5 Run9 关键路径

正式输出：

`experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing\results\semantic_multitask_lesion_balanced_label_smoothing\skin7_yolo26s_sem_lesion_balanced_ls005_seed42`

配置与协议：

- `experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing\train_yolo26s_sem_cls_lesion_balanced_label_smoothing_v9.yaml`
- `experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing\RUN9_PROTOCOL_2026-08-01.md`
- `experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing\VERIFICATION_2026-08-01.md`

关键权重：

- `weights\best_macro_f1.pt`：epoch 49，53,319,129 bytes；
- `weights\best_feasible_macro_f1.pt`：epoch 37，53,316,953 bytes；
- `weights\last.pt`：正常早停后的最终权重，13,440,031 bytes；
- `weights\epoch*.pt`：每 5 个 epoch 的周期快照，保留训练中间状态。

`best_macro_f1.pt` 不能因为名字有 best 就自动视为正式胜者；它违反预注册约束。`best_feasible_macro_f1.pt` 是 Run9 内部合规候选，但尚不能替代 Run8/Run4。

---

## 11. 历史基线与当前项目仍存在的主要缺点

### 11.1 多任务负迁移没有解决

Run5~Run9 的共同模式是：分类提高时，前景分割、mel→nv 或 nv→mel 至少有一项恶化。继续堆叠注意力模块的边际价值已经很低。

### 11.2 有效样本量远小于图片数量

训练集 8512 张中有 2997 张离线增强副本。df、vasc、akiec 的独立病灶很少，模型容易记忆病灶纹理模板。

该描述针对旧 `processed_skin7_v3` 和 Run1~Run9。新 IMA++ 二值分割集不含这些旧离线增强副本，但它本身不解决七类疾病分类的长尾问题。

### 11.3 单一 HAM10000 域和来源混杂

类别与医院、设备、色彩、裁剪及诊断确认方式相关。模型可能学习来源特征，而不是疾病特征。

### 11.4 掩码 provenance 不完整

官方 ISIC 2018 分类任务和分割任务不是同一规模。旧 V3 历史数据制作曾假设大量 HAM10000 图片都有掩码，但项目文档没有为每个旧掩码保存权威来源、人工/自动属性和版本信息。该风险仍适用于旧 V3，不得反向补写为已验真；新的 IMA++/ISIC 2018 数据则已经保存 mask source、工具、标注技能、MD5/SHA、许可和版本信息，provenance 门槛已通过。

### 11.5 当前“七类逐像素语义”不是独立像素病理标注

疾病标签是把图像诊断广播到病灶区域得到的。这会混淆定位和诊断目标，是当前首要任务定义风险。

新 IMA++ 数据已经停止这种广播，只输出二值病灶掩码。该缺陷仍属于 Run4~Run9 的历史八通道基线；新多任务分类头尚待实现。

### 11.6 单 seed 和单 split 结论不稳

核心实验主要使用 seed=42。val 中 df=11、vasc=15 个病灶，单次提升可能是随机波动。

### 11.7 现有 test 已被历史开发间接利用

V3 的建立参考过 V2 test 下降，因此不能再把当前 test 描述成完全未见终极盲测集。最终需要真正外部验证或新的锁定测试集。

### 11.8 尚未完成可靠性工作

- probability calibration；
- ECE/Brier score；
- 多随机种子；
- lesion/patient 级交叉验证；
- 外部医院/设备验证；
- 肤色公平性；
- hard-case 和异常 mask 人工审计；
- 最终模型的统一 test。

### 11.9 早期实例分割与当前语义基线不完全同构

Run3→Run4 同时改变了 YOLO 版本、模型头和任务形式。因此不能把所有提升归因于 C2PSA 或语义分割。

---

## 12. 新数据集与语义分割兼容性结论

### 12.1 已确认不能直接用于当前语义训练的数据

- **ISIC 2019 / BCN20000**：只有图像级疾病诊断和元数据，没有官方像素级病灶掩码；不能直接训练当前语义分割头。
- **PAD-UFES-20、DDI**：主要是普通临床/手机照片，模态与当前皮肤镜数据不同，也不提供当前七类逐像素语义标签。

### 12.2 能支撑病灶语义分割、但不能直接提供七病种像素标签的数据

- **IMA++**：已下载并完成主数据构建。14,967 张原图、22,472 张原始掩码中，v1 选出 11,409 对高置信二值语义样本；多人标注使用 STAPLE，单人标注只纳入 S1，输出目录为 `Data\processed_imaplusplus_binary_v1`。
- **ISIC 2018 Task 1**：2,594 对原图/病灶掩码已完整配对，但 2,505 个 ISIC ID 与 IMA++ 重叠；严格去重后只有 89 对增量，已存入 `Data\processed_isic2018_task1_incremental_v1`，只作外部候选。
- **ISIC 2018 Task 2**：已下载的 12,970 张图对应 5 类皮肤镜属性掩码，不是病灶/背景主掩码，本轮没有混入二值真值。
- **PH2**：有人工病灶分割和有限诊断类别，适合二值分割或外部验证，不适合作为七类主训练集。

### 12.3 2026-08-04 已验证的数据事实

- IMA++ 输出 train/val/test=`10483/292/634`，磁盘掩码值域严格为 `{0,1}`；
- 3,556 个 S2 非专家单标注样本未进入首个干净基线；
- 1 个空掩码和 1 个全图掩码已隔离；
- 136 个极端面积或多连通域样本保留并写入 `quality_flags.csv`，没有静默修改官方边界；
- 官方图像级 split 在严格分组后出现 71 个锚点冲突，已按 `test > val > train` 整组迁移；
- train/val/test 的非空 patient_id、lesion_id 两两交集为 0；10,595 个自动关联组均只属于一个 split；
- 当前 Ultralytics loader 已实际读取 train、val、test、external 样本；磁盘 0/1 掩码经 letterbox 后合法产生 `255=ignore padding`；
- 构建外的全量配对/尺寸/值域检查和每个数据集 128 文件哈希抽样通过；
- 没有运行模型 test 指标。

### 12.4 正确的数据使用方向

如果采用二值语义分割 + 分类头：

- `processed_imaplusplus_binary_v1` 用于主要分割监督；
- ISIC 2018 的 89 对增量只作补充外部候选，不进入训练；
- ISIC 2019 去除 HAM10000 重复后用于分类监督；
- 同时有两种标签的交集用于联合监督；
- 只有一种标签的样本使用 masked/partial multitask loss。

如果坚持当前八通道疾病语义头：

- 只能使用“可信二值掩码 + 七类诊断”精确匹配的 ID 交集；
- 必须记录掩码来源、诊断来源、license、patient/lesion/source；
- 不能把完整 ISIC 2019 直接并入分割训练。

---

## 13. 项目当前正在执行的计划

当前阶段不是训练阶段，而是 **数据和任务定义治理阶段**：

1. **已完成**：冻结 experiments 01~09、V3 数据和现有 test；
2. **已完成**：IMA++/ISIC 2018 原始文件、掩码来源、ID、许可和元数据审计；
3. **已完成**：IMA++ 二值分割数据构建、patient/lesion/hash/感知重复分组和机器可读报告；
4. **已完成**：ISIC 2018 Task 1 与 IMA++ 精确交集及 89 对增量候选构建；
5. **已完成**：二值语义数据 loader 和全量完整性验收；
6. **下一步**：锁定分类 taxonomy，并审计 ISIC 2019/其他官方分类数据的真实增量；
7. **下一步**：实现二值分割 + 图像分类双头及 partial-label loader；
8. **下一步**：重新定义新基线的 val 指标、风险约束和训练协议；
9. **最后**：通过项目级启动门槛并获得用户确认后，把 Run10 交给训练 AI。

---

## 14. 训练启动门槛

Run10 开始前必须全部满足：

- [x] IMA++/ISIC 2018 中每个拟使用掩码均有可追踪来源和版本；
- [ ] 每个疾病标签均有明确 taxonomy 映射；
- [x] 分割数据内部及 ISIC 2018 增量已按 ISIC ID、SHA-256 和严格感知 hash 去重；分类数据尚未加入；
- [x] IMA++ 的 patient/lesion/hash 分组规则已锁定；真正外部域仍缺失 patient/lesion 信息；
- [ ] IMA++ val/test 角色已锁定，ISIC 2018 89 对角色已锁为外部候选；真正独立外部测试仍未确定；
- [x] 当前模型 test 未被新数据或模型决策使用；
- [ ] 新分割标签已锁定为二值，但图像分类头、taxonomy 和是否保留历史八通道对照尚未形成最终实验协议；
- [ ] 二值 SemanticDataset loader 已通过；partial-label 多任务 loader 尚未实现；
- [x] 新二值数据文件数、mask 值域、空/全图、异常轮廓、许可和 overlap 审计已通过；
- [ ] 新模型选择指标和约束预注册；
- [x] 项目根目录存在 `DATA_TASK_ACCEPTANCE_2026-08-04.md`；
- [ ] 最新文档明确写出 `PROJECT_RUN10_GATE=PASS`；当前仍为 `BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`；
- [ ] 用户确认可以开始训练。

任一项未满足时，训练 AI 应停止在 dry-run/设计阶段，不得擅自启动长训练。

---

## 15. 未来路线规划

### 阶段 A：数据来源和标签审计

状态：**二值分割部分已完成。**

- IMA++/ISIC 2018 官方源文件保存在 `Data\1`，没有删除或原地修改；
- `manifest.csv` 已建立 image、lesion、patient、diagnosis、mask_source、license 和 hash 关系；
- STAPLE/S1/S2、空/全图和质量标记策略已落盘；
- `prepare_imaplusplus_binary_v1.py`、`prepare_isic2018_task1_incremental_v1.py` 和验收报告已落盘；
- 分类数据来源和 taxonomy 审计仍未完成。

### 阶段 B：任务结构锁定

状态：**标签方向已明确，模型协议未完成。**

推荐首先实现并比较：

1. 二值语义分割 + 七类图像分类的干净多任务基线；
2. 当前八通道疾病语义基线，作为历史对照；
3. 两者使用相同 backbone、输入、数据交集和训练预算。

如果二值分割+分类明显减少负迁移，它应成为后续论文主线。

### 阶段 C：构建 V4

状态：**二值分割数据版本已完成，完整多任务 V4 尚未完成。**

- 已建立 `processed_imaplusplus_binary_v1`，不含旧 V3 离线增强副本；
- 已建立不可变 manifest、逐文件 hash、版本号、质量报告和 `_SUCCESS`；
- 已保留 IMA++ val/test 和 ISIC 2018 external candidate 角色；
- 分类样本、partial-label loss、分类平衡和 source-held-out 方案仍待实现。

### 阶段 D：Run10 新基线

Run10 应被视为“任务/数据重定义后的新基线”，不能假装与 Run9 只差一个小参数。必须：

- 使用新实验目录；
- 记录初始化权重；
- 先 dry-run，再 1% 冒烟，再正式训练；
- 完成断点恢复冒烟；
- 只用 val 选择模型；
- 与 Run4、Run8、Run9 按统一口径比较。

### 阶段 E：注意力与采样消融

在干净新基线上逐个测试：

- 无注意力基线；
- 标准 CBAM/ECA 对照；
- 病灶引导注意力；
- lesion-balanced sampler；
- label smoothing/其他分类正则；
- 每次只改变一个因素。

### 阶段 F：稳定性、校准和外部验证

- 至少 3 个 seed，或 lesion 级多折交叉验证；
- bootstrap CI；
- temperature scaling、ECE、Brier；
- PH2/外部 ISIC/source-held-out 验证；
- 失败案例和异常 mask 人工审查；
- 方法完全冻结后，经用户明确授权运行最终 test。

### 阶段 G：论文产出

- 数据流图；
- 模型结构图；
- 每轮唯一变量和消融表；
- 图像级/病灶级混淆矩阵；
- 分割可视化和 hard cases；
- 参数/GFLOPs/速度表；
- 置信区间与校准图；
- 数据局限、域偏差和临床适用边界。

---

## 16. 下一位训练 AI 的执行纪律

1. 先读本文件、最新 resolution 报告和目标实验协议。
2. 在最新文档明确写出 `PROJECT_RUN10_GATE=PASS` 且用户确认前不启动 Run10；`BINARY_SEGMENTATION_DATA_GATE=PASS` 不能替代它。
3. 正式训练优先从 PyCharm 可见启动。
4. 每个实验创建独立 `experiments\10_...`、`11_...` 目录。
5. `exist_ok` 应保持 false 或使用新的唯一 run name，防止覆盖。
6. 每次实验只改变预注册的一个因素；若任务/数据一起改变，明确称为新基线而非消融。
7. 正式训练必须保存 `last.pt`、`best.pt`、专项 best、周期 checkpoint、args、results.csv 和图表。
8. 中断恢复必须恢复模型、EMA、optimizer、scheduler、scaler、epoch 和早停/选择状态；不能只加载权重后冒充 resume。
9. resume 前先运行安全 dry-run，拒绝不完整或已被 strip 的 checkpoint。
10. 不要从正常早停后被 optimizer-stripped 的 `last.pt` 强行继续同一正式实验。
11. 不访问 test；任何 test 请求必须回到用户确认。
12. 不删除或清理 01~09、V3、备份、历史报告。
13. 不把 smoke 结果当正式结果，不把训练日志峰值当统一评价结果。
14. 不把实例分割 Mask mAP 与语义分割 mIoU 直接比较。
15. GPU 结论必须写明实际 CUDA 环境和是否真的完成 GPU forward/backward。

---

## 17. 当前可运行入口

### 17.1 Run9 训练入口

```powershell
& '.\.venv\Scripts\python.exe' '.\tools\train_skin7_yolo26_label_smoothed.py' --dry-run
```

正式配置默认指向：

`experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing\train_yolo26s_sem_cls_lesion_balanced_label_smoothing_v9.yaml`

Run9 已正常结束，不应再启动同名正式训练。

### 17.2 Run9 安全恢复审计入口

```powershell
& '.\.venv\Scripts\python.exe' '.\tools\resume_yolo26_label_smoothed_v9.py' --dry-run
```

该脚本默认检查 Run9 `last.pt`。由于 Run9 已正常早停，当前用途是审计而不是继续训练。

### 17.3 统一 val 评价入口模板

```powershell
& '.\.venv\Scripts\python.exe' '.\tools\evaluate_skin7_unified.py' `
  --weights '<checkpoint.pt>' `
  --split val `
  --decision classifier `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --output-dir '<new-val-output-dir>'
```

禁止在未获得用户授权时添加 `--split test --allow-test`。

---

## 18. 关键脚本索引

### 数据

- `tools\prepare_skin7_dataset.py`：V1 数据构建历史脚本；
- `tools\prepare_skin7_dataset_v2.py`：V2；
- `tools\prepare_skin7_dataset_v3.py`：V3；
- `Data\processed_skin7_v3\metadata\train.csv`：训练 lesion/source 元数据；
- `Data\processed_skin7_v3\metadata\v3_audit.json`：V3 受控变量审计。

### 语义训练与评价

- `tools\train_skin7_yolo26_sem.py`：Run4 语义训练；
- `tools\evaluate_skin7_unified.py`：统一评价；
- `experiments\04_YOLO26s_Semantic_C2PSA\data_semantic_eval_v4.yaml`：正确 8 类 standalone 评价配置。

### 多任务与注意力

- `tools\skin7_multitask.py` / `train_skin7_yolo26_multitask.py`：Run5；
- `tools\skin7_cbam.py` / `train_skin7_yolo26_cbam.py`：Run6；
- `tools\skin7_lesion_guided.py` / `train_skin7_yolo26_lesion_guided.py`：Run7；
- `tools\skin7_lesion_balanced.py` / `train_skin7_yolo26_lesion_balanced.py`：Run8；
- `tools\skin7_label_smoothed.py` / `train_skin7_yolo26_label_smoothed.py`：Run9。

### 恢复

- `tools\resume_yolo26_sem_v4.py`；
- `tools\resume_yolo26_multitask_v5.py`；
- `tools\resume_yolo26_cbam_v6.py`；
- `tools\resume_yolo26_lesion_guided_v7.py`；
- `tools\resume_yolo26_lesion_balanced_v8.py`；
- `tools\resume_yolo26_label_smoothed_v9.py`。

---

## 19. 明确禁止事项

- 不覆盖或重命名 experiments 01~09；
- 不把 Run9 epoch 49 直接宣布为最终模型；
- 不在前置数据/任务门槛通过前开始 Run10；
- 不把 `BINARY_SEGMENTATION_DATA_GATE=PASS` 误读成整个 `PROJECT_RUN10_GATE=PASS`；
- 不使用 ISIC 2019 直接训练语义分割头，因为其官方版本没有像素级掩码；
- 不把与 IMA++ 重叠的 2,505 对 ISIC 2018 图像当作外部验证；
- 不把 ISIC 2018 Task 2 的五类属性掩码当作 Task 1 病灶主掩码；
- 不把 89 对 ISIC 2018 增量候选并入训练；
- 不把未知来源掩码当人工金标准；
- 不把同一 HAM10000 图在不同数据包中的副本当新样本；
- 不使用图片级随机划分破坏 lesion/patient 安全；
- 不根据 test 选择数据、损失、阈值或模型；
- 不从备份 Move/Cut；
- 不做破坏性清理；
- 不把当前模型描述为临床可用产品。

---

## 20. 交付完成定义

下一阶段训练工作只有在以下成果均落盘后才算完成：

- 新实验协议；
- dry-run 记录；
- CUDA 冒烟记录；
- 正式训练配置；
- 完整训练日志与周期 checkpoint；
- 正常早停/中断恢复分析；
- standalone unified val 结果；
- 与 Run4、Run8、Run9 的公平对比；
- 未访问 test 的明确声明；
- 更新后的项目索引和交接文档。

---

## 21. 可直接发给下一位训练 AI 的开场提示

```text
请先完整阅读项目根目录 PROJECT_HANDOFF_2026-08-03.md 和 DATA_TASK_ACCEPTANCE_2026-08-04.md。
项目路径是 E:\App\DaiMa\JetBrains\Project\PiFuV2。
Run1~Run9 已完成并必须保留；Run9 在 epoch 69 正常早停。
旧 V3 只用于历史复现。新的 IMA++ 二值分割数据位于 Data\processed_imaplusplus_binary_v1，train/val/test=10483/292/634，BINARY_SEGMENTATION_DATA_GATE=PASS。
ISIC 2018 Task 1 与 IMA++ 重叠 2505 个 ID，严格去重后 89 对位于 Data\processed_isic2018_task1_incremental_v1，只作外部候选、不得训练。
你负责后续模型实现、训练、断点恢复、监控和 val 统一评价，不得改动新数据 manifest、split、掩码值域或 test 规则。
当前 PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT。先完成分类 taxonomy、官方分类数据去重、partial-label loader、多任务模型和新指标协议。
只有最新文档明确写出 PROJECT_RUN10_GATE=PASS 且用户确认后，才设计并启动 experiments\10_...。
正式模型选择前禁止运行 test。每次实验只改变预注册因素，正式训练从 PyCharm 启动，所有结果写入新的 experiments 编号目录。
```

---

## 22. 最终判断

九轮实验已经证明：继续微调注意力、采样或分类正则能够改变局部指标，但不能稳定同时改善疾病分类、前景分割和 mel→nv 风险。2026-08-04 已经解决了新二值分割数据的来源、规范掩码选择、客观异常隔离、重复泄漏和 loader 契约问题；当前首要瓶颈进一步收敛为“分类 taxonomy、部分标签多任务实现、独立外部域和新模型选择协议”。

因此正确顺序是：

1. 保持已验收的 IMA++ 二值分割数据不变；
2. 完成分类 taxonomy、分类数据去重和 partial-label 多任务实现；
3. 预注册新指标与约束，通过 `PROJECT_RUN10_GATE` 并取得用户确认；
4. 再由下一位训练 AI 建立 Run10 新基线；
5. 在新基线上做严格单变量消融；
6. 最后做多 seed、校准、真正外部验证和一次性 test。

任何跳过前置治理、直接开始第十次长训练的做法，都很可能重复 Run5~Run9 已经暴露的问题。
