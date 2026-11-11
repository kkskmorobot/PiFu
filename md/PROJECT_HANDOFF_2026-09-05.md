# PiFuV2 项目完整交接文档（Run26 完成后）

更新时间：2026-09-05  
项目根目录：`E:\App\DaiMa\JetBrains\Project\PiFuV2`  
交接目的：把 Run26 之后的模型设计、预检、正式训练、断点恢复、监控、统一 val 评价、最终模型锁定和论文整理交给下一位 AI。  
事实来源：项目磁盘中的配置、manifest、训练 CSV、完成标记、checkpoint manifest、历史结果文档，以及本次只读复核。  

`PROJECT_CURRENT_STATE=RUN26_COMPLETE_HANDOFF_READY`

`RUN26_EXECUTION_STATUS=PASS_NORMAL_EARLY_STOP`

`RUN26_CHECKPOINT_INTEGRITY=PASS`

`RUN26_FORMAL_MODEL_LOCK=NOT_CREATED_NO_FEASIBLE_CHECKPOINT`

`NEXT_TRAINING_GATE=BLOCKED_PENDING_NEW_RUN_PREREGISTRATION_PREFLIGHT_AND_USER_CONFIRMATION`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

---

## 0. 下一位 AI 必须先读的最短结论

1. 本项目是大学毕业论文项目，题目方向是**基于注意力机制改进 YOLO 的皮肤病识别算法研究**。当前主任务是二值病灶语义分割与八类皮肤病图像分类的 partial-label 多任务学习。
2. Run1~Run26 的正式结果都必须保留，不得覆盖、重命名或删除。Run17 的边界注意力目录是未训练的废弃草案，正式 Run17 是无注意力 YOLO26m baseline。
3. Run26 已于 2026-09-04 19:54:32 +0800 正常早停完成。CSV 有连续 epoch 1~71 共 71 行，非有限值 0；不是崩溃。
4. Run26 最佳联合点是 checkpoint bundle `epoch_0040`，对应可见 CSV epoch 41。该点 Binary IoU=0.76971、Image Macro-F1=0.55333、Group Macro-F1=0.63762、Group Macro-Recall=0.65132、MEL→NV=0.33673、Joint=0.70056。
5. Run26 相对同 seed44 的 Run25 ECA-only，在最佳联合点上 Joint +0.01007、Group Macro-F1 +0.01391、Image Macro-F1 +0.02572、Binary IoU +0.00530，并把 MEL→NV 降低 0.11225。相对 Run24 plain，Joint +0.01993、Group Macro-F1 +0.03121。
6. 因而 Run26 是当前很有前景的轻量注意力结果，但只有 seed44 一组，尚不能宣称稳定有效。下一步优先补 seed42/43 的同结构严格配对复现，而不是立即堆新模块。
7. Run26 仍有 0 个正式 feasible checkpoint：MEL→NV=0.33673 远高于锁定上限 0.1077844311。因此 `best_joint_score` 只是 development-val 诊断点，不能冒充正式锁定模型。
8. 当前没有启动下一轮训练。新一轮必须创建新的 `tools\runXX` 和 `experiments\XX_...`，完成预注册、静态/GPU/恢复预检、发布门和用户明确确认后，才可从 PyCharm 启动。
9. 正式模型锁定前禁止运行 IMA++ test、PAD-UFES-20 或 ISIC 2018 external。它们不得用于模型选择、阈值、温度、损失或回训。
10. 不得改动已冻结的新数据 manifest、split、掩码值域、taxonomy 或 test 规则。

---

## 1. 整体任务目标

### 1.1 工程目标

构建一个可复现、可恢复、可审计的医学图像多任务模型：

- 输入皮肤镜或皮肤病灶图像；
- 输出病灶/背景二值像素级分割；
- 输出八类皮肤病图像级诊断；
- 允许一部分样本只有分割标签、另一部分只有分类标签；
- 在长尾分类下提高 group/lesion 级 Macro-F1 和 Macro-Recall；
- 重点降低黑色素瘤被误判为普通痣的 `mel→nv` 风险；
- 完整保存 model、EMA、optimizer、scheduler、AMP scaler、早停、选择状态、RNG 和数据顺序，支持安全中断后真正恢复；
- 所有模型选择只使用冻结的 development val；
- 每次正式实验只改变一个预注册因素。

### 1.2 论文目标

论文需要回答的核心问题不是“哪次训练 Accuracy 最高”，而是：

1. 注意力模块在控制主干、数据、损失、优化器、训练预算和随机种子后，是否稳定改善皮肤病识别？
2. 通道注意力、空间注意力和病灶引导注意力分别带来什么净效应？
3. 改进是否同时保持二值分割质量，而不是以牺牲定位换分类？
4. 单 seed 的表面提升是否大于随机种子波动？
5. 长尾分类和 `mel→nv` 风险能否在不破坏整体性能的情况下改善？
6. 最终模型在锁定的 internal blind test 和外部域上是否仍然成立？

### 1.3 最终交付目标

最终论文至少应包含：

- Binary IoU、Dice、pixel accuracy；
- image-level 与 group/lesion-level Accuracy、Macro-Precision、Macro-Recall、Macro-F1；
- 八类逐类 precision/recall/F1/support；
- `mel→nv`、`nv→mel`、malignant→benign 风险；
- NLL、multiclass Brier、ECE；
- group/lesion 分层 bootstrap 95% CI；
- 至少三 seed 的均值、标准差和配对差值；
- 参数量、GFLOPs、RTX 5080 推理速度和显存；
- 分割图、混淆矩阵、注意力可视化和失败案例；
- 方法完全冻结后一次 internal test，以及获得用户授权后的外部候选评价；
- 数据偏差、标签限制、外部域差异与“非临床产品”边界。

---

## 2. 项目类型，以及为什么这样定义

本项目同时属于：

- 医学图像分析；
- 皮肤病/皮肤镜图像分类；
- 二值语义分割；
- partial-label 多任务学习；
- 长尾类别学习；
- 基于注意力机制的 YOLO 改进；
- 风险敏感模型选择与外部泛化研究。

旧 Run4~Run9 把图像级七病种诊断广播到整个病灶区域，形成“七病种像素类别+背景”的八通道语义头。这不是真正的逐像素病理标签，容易把病灶定位和疾病诊断压在同一个头上，造成多任务负迁移。

Run10 起已经解决该任务定义问题：

- 分割头固定为单通道 lesion logit；
- 磁盘 mask 固定为 `0=background, 1=lesion`；
- 图像分类头独立输出八类 logits；
- 只有分割标签的样本只计算分割 loss；
- 只有分类标签的样本只计算分类 loss；
- classification-only 样本的全 255 mask 只是 ignore 占位，不是伪病灶真值。

保留分割任务是必要的，因为论文不仅要回答“是什么病”，也要回答“病灶在哪里”，同时病灶边界和空间结构正是注意力机制的研究依据。

本项目当前只可描述为辅助识别算法研究，不得宣称是可直接用于临床诊断的产品。

---

## 3. 已冻结的数据与标签契约

### 3.1 IMA++ 二值分割主数据

目录：`Data\processed_imaplusplus_binary_v1`

- 总计 11,409 对有效图像/掩码；
- train/val/test=`10,483/292/634`；
- 磁盘掩码严格为 `{0,1}`；
- letterbox 后的 `255` 仅表示 ignore padding；
- 多标注样本使用 STAPLE 共识；
- 单标注干净基线只使用专家 S1；
- 3,556 个 S2 非专家单标注未进入 v1；
- 1 个空掩码和 1 个全图掩码已隔离；
- 136 个极端面积/多连通域样本保留并显式写入质量标记；
- patient、lesion、SHA-256、严格 pHash+dHash+尺寸已经绑定防泄漏组；
- train/val/test 的非空 patient_id、lesion_id 交集均为 0；
- `BINARY_SEGMENTATION_DATA_GATE=PASS`。

禁止重新划分、重写 manifest、修改掩码值域或调整 test 角色。

### 3.2 ISIC 2019 增量分类数据

目录：`Data\processed_isic2019_classification_incremental_v1`

- 官方 training 原始 25,331；
- 排除 HAM10000 ID 10,015、IMA++ 原始 ID 828，并完成文件/感知重复审计；
- 最终严格增量 14,431；train/val=`12,979/1,452`；
- classification-only mask 全 255，仅作 ignore 占位；
- split 按 lesion ID、SHA-256 和严格感知签名绑定，seed=42；
- 1,256 个缺失 lesion ID 的样本只能视为 singleton，这是仍需在论文中报告的限制。

### 3.3 当前联合 development manifest

Run10~Run26 使用的冻结联合数据规模：

| split | 总样本 | 分割监督 | 分类监督 | 双监督 |
|---|---:|---:|---:|---:|
| train | 23,462 | 10,483 | 13,715 | 736 |
| val | 1,744 | 292 | 1,488 | 36 |

每个样本显式包含 `has_segmentation`、`has_classification`、`classification_target`、`dataset_role`、`split`、`group_id` 和 `isic_id`。

### 3.4 八类分类 taxonomy

顺序已冻结，不得调整：

| ID | 类别 | 含义 |
|---:|---|---|
| 0 | akiec | 光化性角化病/鲍恩病相关病变 |
| 1 | bcc | 基底细胞癌 |
| 2 | bkl | 良性角化性病变 |
| 3 | df | 皮肤纤维瘤 |
| 4 | mel | 黑色素瘤 |
| 5 | nv | 黑色素细胞痣 |
| 6 | vasc | 血管性病变 |
| 7 | scc | 鳞状细胞癌 |

机器可读文件：`tools\classification_taxonomy_v1.yaml`。

### 3.5 永不用于当前开发训练的数据

- `Data\processed_imaplusplus_binary_v1\test`：634 对，internal blind test；
- `Data\processed_isic2018_task1_incremental_v1`：严格去重后的 89 对，只作 external segmentation candidate；
- `Data\processed_pad_ufes_20_external_classification_v1`：2,269 张，只作 external classification candidate；
- ISIC 2019 challenge test：不得读取或利用。

ISIC 2018 Task 1 的 2,594 对中有 2,505 个 ID 与 IMA++ 重叠，因此完整 ISIC 2018 不能被包装成独立外部集。PAD-UFES-20 是手机临床图像域，只覆盖八类中的六类，也不能替代 internal blind test。

---

## 4. 当前模型与训练合同

### 4.1 Run18 以后共同基线

- 正式主干：本地 `models\yolo26m-sem.pt`；
- AMP 自检小模型：本地 `tools\yolo26n.pt`，不是正式训练主干；
- task：YOLO semantic backbone + 二值分割头 + 独立八分类头；
- batch=16、imgsz=640、workers=8；
- epochs=100、AdamW、AMP、确定性 seed；
- lr0=0.001、lrf=0.01、cosine schedule、weight decay=0.0005；
- 分类 loss：unweighted cross entropy，label smoothing=0；
- image classification loss weight=0.5；
- 每轮 23,462 个 train 样本无放回打乱一次；
- Mosaic/MixUp/CutMix/CopyPaste 均关闭，以保持单图分类标签语义；
- test 不进入配置。

### 4.2 修正后的延迟效用门控早停

Run17 在可见 epoch 26 就早停，stopper 最佳点停在 epoch 6，分类尚未成熟。该生命周期问题已由 Run18 解决：

- 可见 epoch 1~39 不累计 patience；
- 从 Trainer epoch 40 后启用 patience=30；
- 最早只能在约可见 epoch 70 停止；
- 未达到 Binary IoU≥0.70 且 Group Macro-F1≥0.50 时，early-stop fitness=`joint_score-2`；
- 达到效用门后，fitness 使用原 joint 减 MEL→NV 超限惩罚；
- 指标本身和 checkpoint 选择别名不变。

Run18 完整训练 100 epochs 后明显优于 Run17，证明该修正必要。Run18~Run26 均继承这一生命周期。

### 4.3 checkpoint 合同

正式 bundle 版本为 `pifuv2_multitask_checkpoint_v2`，必须保存并恢复：

- live model 和 EMA；
- optimizer、scheduler、AMP scaler；
- completed epoch/global step；
- stopper 与专项模型选择状态；
- Python/NumPy/Torch CPU/CUDA RNG；
- sampler、loader generator 和数据进度；
- 配置、代码、数据和模型权重指纹。

只有 manifest 中大小与 SHA-256 全部匹配且存在 `_SUCCESS` 的 bundle 才能恢复。正常完成/正常早停后不得继续写入同名正式实验。

### 4.4 模型选择规则

- `joint_score=sqrt(Binary IoU × Group Macro-F1)`；
- 效用门：Binary IoU≥0.70 且 Group Macro-F1≥0.50；
- 风险门：MEL→NV≤0.1077844311；
- 主候选别名：`best_feasible_joint`；
- 无可行点时必须登记 constraint failure，只可把 `best_joint_score` 当诊断点。

注意：checkpoint 代码中的 `best_macro_f1` 实际跟踪 `metrics/group_macro_f1`，不是 `metrics/image_macro_f1`。阅读别名时必须结合代码和 CSV，不能仅按文件名猜测。

---

## 5. 已完成的工作

### 5.1 数据与任务治理

以下原先阻塞 Run10 的问题均已解决，不需要下一位 AI 重做：

- 分类 taxonomy 已锁为八类；
- ISIC 2019 官方分类数据完成 ID、文件哈希和严格感知重复去重；
- 联合 development manifest 与 split 已冻结；
- partial-label loader 已实现并验收；
- 二值分割头 + 独立分类头已实现；
- partial loss 的有/无标签边界已验证；
- val-only 统一指标协议已实现；
- 完整 checkpoint、自动恢复和指纹合同已实现；
- CPU 与 RTX 5080 GPU/AMP/workers 恢复一致性已经多轮验证；
- PAD-UFES-20 和 ISIC 2018 external 数据角色已锁定。

### 5.2 工程故障与整理问题

以下历史问题也已经解决，不应在后续误判为当前阻塞项：

- Run11 入口曾出现尾部误字符导致语法错误，正式训练前已修复；
- Run12 的用户确认发布字段已经正确记录；
- Run14 首次在 epoch 1 完成前因断电中止，半成品已归档，development manifest 通过确定性投影恢复，随后 fresh start 并正常完成；
- Run18 曾被误建成普通文件，已备份处理并正确建立 `tools\run18` 目录；
- Run20 目录曾被 PyCharm 项目视图误删，已恢复并完成正式训练；
- Run24 配置 SHA-256 文档中的单字符抄写错误已更正，正式训练随后完整完成；
- Run17 过早早停的设计缺陷已由 Run18 的延迟效用早停解决；
- 根目录散乱的 Python/YAML 已按用户要求整理进 `tools\run1` 到 `tools\run26`；未来 Run27 应放入 `tools\run27`，而不是重新把入口散落到根目录。

网络慢、账户余额、熄屏或 PyCharm 视图状态都不是当前模型或数据合同的一部分，不需要写入后续训练设计。

### 5.3 正式实验完成度

- Run1~Run9：旧 V3 历史体系，全部保留，仅用于复现与历史讨论；
- Run10~Run16：新 IMA++/ISIC 2019 partial-label 双任务体系的 YOLO26s→YOLO26m 探索；
- Run17~Run18：YOLO26m plain baseline 与早停生命周期修正；
- Run19~Run20：病灶引导和自适应残差病灶引导注意力消融；
- Run21~Run25：P3/P4 ECA-only 的三 seed 严格配对；
- Run26：在 seed44 ECA 后增加恒等初始化空间门的单因素实验，已经正常完成。

---

## 6. 关键实验结果

### 6.1 Run1~Run9：旧 V3 历史阶段

| Run | 主要变化 | 关键结论 |
|---:|---|---|
| 1 | YOLO11s-seg V1 | 初始实例分割基线，Mask mAP50-95 峰值约 0.5801 |
| 2 | 更强数据平衡与 hard-example 增强 | epoch 58 早停，Mask mAP50-95 降至约 0.5306 |
| 3 | 稳定增强、NV cap=3200 | Mask mAP50-95≈0.5752，图像 Macro-F1≈0.6994 |
| 4 | YOLO26s semantic + 原生 C2PSA | 前景 mIoU=0.538748，是旧任务最强分割基线 |
| 5 | P3/P4 全局池化辅助分类头 | 多任务负迁移，分割降至 0.470054 |
| 6 | Run5 + 标准 CBAM | 少数类 recall 改善，但分割和风险未同时改善 |
| 7 | P3/P4/P5 病灶引导双注意力 | 没有稳定全面优势 |
| 8 | lesion-balanced sampler | 首次获得旧协议合规 checkpoint；epoch79 Macro-F1 0.712415 但风险不合规 |
| 9 | Run8 + label smoothing=0.05 | epoch69 正常早停；分类峰值提高但分割/风险冲突仍在 |

Run1~Run3 的实例分割 Mask mAP 与 Run4~Run9 的 semantic mIoU 不是同一指标，禁止直接横向比较。

### 6.2 Run10~Run16：新 partial-label 多任务体系

| Run | 唯一因素/结构 | epochs | 峰值 Binary IoU | 峰值 Image Macro-F1 | 峰值 Group Macro-F1 | 峰值 Joint | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 10 | YOLO26s plain 新基线 | 100 | 0.78131 | 0.54293 | 0.59426 | 0.68132 | 完整基线，无可行点 |
| 11 | MEL target CE 权重 1.81785 | 83 | 0.77894 | 0.51780 | 0.57017 | 0.66254 | 风险方向改善但整体下降，仍未过门 |
| 12 | YOLO26s BLG-MDA | 91 | 0.77914 | 0.55403 | 0.59969 | 0.68258 | 小幅正增益，风险仍失败 |
| 13 | Run12 + MEL weighted CE | 72 | 0.77632 | 0.53763 | 0.59183 | 0.67586 | 两因素不利交互 |
| 14 | MEL-vs-NV logit margin | 62 | 0.77703 | 0.53792 | 0.58802 | 0.67369 | 仅冷启动风险低，成熟点无效 |
| 15 | pairwise boundary loss | 58 | 0.77587 | 0.54019 | 0.57479 | 0.66724 | 未超过 Run12，不继承该 loss |
| 16 | Run12 主干升级 YOLO26m | 61 | 0.77389 | 0.54501 | 0.60160 | 0.68049 | 有效注意力版本，但缺同主干 plain 对照 |

这些列是对应指标的独立峰值，未必来自同一 epoch。Run10~Run16 均无正式 feasible checkpoint。

### 6.3 Run17~Run26：YOLO26m 严格注意力消融

下表全部取各 run 的 `best_joint_score` 对应同一 CSV 行，便于公平比较；bundle epoch 是零基，表内 epoch 是可见的一基 CSV epoch。

| Run | seed | 结构 | epochs | Joint epoch | Binary IoU | Image Macro-F1 | Group Macro-F1 | Group Recall | MEL→NV | Joint |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 42 | plain，旧早停 | 26 | 23 | 0.75747 | 0.47692 | 0.52180 | 0.49733 | 0.46939 | 0.62869 |
| 18 | 42 | plain，延迟效用早停 | 100 | 50 | 0.76323 | 0.54417 | 0.61023 | 0.59915 | 0.38776 | 0.68246 |
| 19 | 42 | 固定 BLG-MDA | 71 | 36 | 0.76971 | 0.54178 | 0.60160 | 0.59004 | 0.37755 | 0.68049 |
| 20 | 42 | 自适应残差 BLG | 100 | 53 | 0.76327 | 0.50922 | 0.57874 | 0.55872 | 0.38776 | 0.66463 |
| 21 | 42 | P3/P4 ECA-only | 100 | 41 | 0.77431 | 0.53745 | 0.60219 | 0.60337 | 0.40816 | 0.68285 |
| 22 | 43 | plain | 93 | 60 | 0.76377 | 0.55017 | 0.62811 | 0.64033 | 0.40816 | 0.69263 |
| 23 | 43 | P3/P4 ECA-only | 70 | 62 | 0.76181 | 0.55045 | 0.58762 | 0.56344 | 0.40816 | 0.66907 |
| 24 | 44 | plain | 100 | 78 | 0.76393 | 0.54301 | 0.60641 | 0.62047 | 0.34694 | 0.68063 |
| 25 | 44 | P3/P4 ECA-only | 73 | 48 | 0.76441 | 0.52761 | 0.62371 | 0.58258 | 0.44898 | 0.69049 |
| 26 | 44 | P3/P4 ECA + spatial7 | 71 | 41 | **0.76971** | **0.55333** | **0.63762** | **0.65132** | **0.33673** | **0.70056** |

Run17 的过早早停使其不能作为成熟对照；正式 plain 生命周期基线从 Run18 开始。Run17~Run26 的正式 feasible checkpoint 数量均为 0。

### 6.4 三 seed ECA-only 结论

ECA−plain 的严格配对结果：

| seed | Δ Binary IoU | Δ Image Macro-F1 | Δ Group Macro-F1 | Δ Group Recall | Δ MEL→NV | Δ Joint |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.01108 | -0.00672 | -0.00804 | +0.00422 | +0.02040 | +0.00039 |
| 43 | -0.00196 | +0.00028 | -0.04049 | -0.07689 | +0.00000 | -0.02356 |
| 44 | +0.00048 | -0.01540 | +0.01730 | -0.03789 | +0.10204 | +0.00986 |
| 平均 | +0.00320 | -0.00728 | -0.01041 | -0.03685 | +0.04081 | -0.00444 |

结论：ECA-only 对 Binary IoU 有极小平均增益，但没有稳定改善分类、group recall、风险或 joint。论文不能宣称 ECA-only 是稳定正改进。

---

## 7. Run26 详细复核

### 7.1 研究因素

Run26 相对 Run25 唯一新增因素：在 P3、P4 的 identity-initialized ECA 输出后分别增加一个 CBAM 风格空间门：

1. 沿通道计算 mean 和 max；
2. 拼成 2 通道空间描述；
3. 使用无 bias 的 7×7 卷积；
4. 通过 `2×sigmoid` 得到乘法门；
5. 卷积全零初始化，使初始门严格等于 1。

该注意力只作用于分类头。原 P3/P4 特征仍直接进入语义分割头；没有预测 mask 引导、没有 P5 描述符、没有修改语义分支。

### 7.2 训练完成与恢复

- 启动记录：2026-09-04 18:06:40 +0800；
- 训练入口：`tools\run26\train_run26_from_pycharm.py`；
- 配置 SHA-256：`60f24b15ec51995f9728e86777bda3dcf6274bb7feac5cfd92652ed7e4500546`；
- 行为/就绪指纹：`a440204a90d71e79e1d1c9cfa25fadcdc7a0c76854db469a01387bae6451abd7`；
- 本次启动从完整 bundle `epoch_0041` 恢复；
- 完成时间：2026-09-04 19:54:32 +0800；
- 完成标记内部 `last_completed_epoch=70`，对应可见 CSV epoch 71；
- CSV epoch 1~71 连续，非有限值 0，错误标记 0；
- 最佳 early-stop fitness 位于可见 epoch 41，之后 30 个 epoch 未刷新，在可见 epoch 71 正常触发 patience；
- 当前没有 Run26 Python 训练进程。

### 7.3 checkpoint 完整性

保留的 5 个 bundle：

- `epoch_0021`：最佳 Binary IoU；
- `epoch_0040`：最佳 Group Macro-F1 与最佳 Joint；
- `epoch_0068`、`epoch_0069`、`epoch_0070`：最后三个完整 epoch。

上述 bundle 的 `trainer.pt`、`runtime_state.pt` 均重新核对：

- `_SUCCESS` 存在；
- 文件字节数与 manifest 一致；
- SHA-256 与 manifest 一致；
- fingerprint 全部等于 `a440204a90d71e79e1d1c9cfa25fadcdc7a0c76854db469a01387bae6451abd7`。

`RUN26_CHECKPOINT_INTEGRITY=PASS`。

### 7.4 Run26 指标

独立峰值：

- 最佳 Binary IoU/Dice：可见 epoch 22，0.77532/0.87344；
- 最佳 raw Image Macro-F1：可见 epoch 70，0.57259；
- 最佳 Group Macro-F1：可见 epoch 41，0.63762；
- 最佳 Group Macro-Recall：可见 epoch 40，0.65132；
- 最佳 Joint：可见 epoch 41，0.70056。

最佳联合 bundle `epoch_0040`（可见 epoch 41）同一行指标：

| 指标 | 数值 |
|---|---:|
| Binary IoU | 0.76971 |
| Binary Dice | 0.86987 |
| Image Macro-F1 | 0.55333 |
| Group Macro-F1 | 0.63762 |
| Group Macro-Recall | 0.65132 |
| MEL→NV | 0.33673 |
| Malignant→Benign | 0.24638 |
| NLL | 0.71640 |
| Brier | 0.35915 |
| ECE | 0.06142 |
| Joint | 0.70056 |

最终可见 epoch 71：Binary IoU=0.76946、Image Macro-F1=0.56919、Group Macro-F1=0.59570、Group Macro-Recall=0.60865、MEL→NV=0.33673、Joint=0.67702。

共有 50 个 utility-eligible epoch，但风险可行 epoch 为 0，完整 feasible epoch 也为 0。全程最低 MEL→NV=0.13265 出现在可见 epoch 1，但该点 utility-eligible=0，属于冷启动状态；成熟可用阶段最低 MEL→NV=0.33673。

### 7.5 相对基线的受控结论

Run26−Run25（同 seed44，空间门的净增益；均取各自最佳 joint 行）：

| 指标 | 差值 |
|---|---:|
| Binary IoU | +0.00530 |
| Image Macro-F1 | +0.02572 |
| Group Macro-F1 | +0.01391 |
| Group Macro-Recall | +0.06874 |
| MEL→NV | -0.11225（更好） |
| Malignant→Benign | -0.05797（更好） |
| NLL | -0.01551（更好） |
| Brier | -0.00244（更好） |
| ECE | +0.00077（基本持平、略差） |
| Joint | +0.01007 |

Run26−Run24 plain（同 seed44 的整体注意力组合效果）：

- Binary IoU +0.00578；
- Image Macro-F1 +0.01032；
- Group Macro-F1 +0.03121；
- Group Macro-Recall +0.03085；
- MEL→NV -0.01021；
- Joint +0.01993。

因此 Run26 的训练是成功的：空间门在 seed44 上同时改善分割、图像分类、group 分类、召回、风险方向和联合指标，没有出现明显交换损失。但它仍只是单 seed 证据，而且 MEL→NV 远未通过正式风险门。

### 7.6 Run26 的正确状态

- 可以写入论文草稿为“有前景的单 seed 空间注意力结果”；
- 不可以写成“空间注意力已经稳定有效”；
- 不可以创建正式模型锁；
- 不可以因为 Joint=0.70056 是当前表中较高值就提前运行 test；
- 不应继续已正常早停的 Run26；
- 后续必须以新编号做 seed42/43 配对复现。

---

## 8. 项目目前的主要缺点与风险

### 8.1 临床风险约束仍未解决

Run26 最佳联合点 MEL→NV=0.33673，约为锁定上限 0.1077844311 的 3.12 倍。当前多个高效用实验都没有成熟风险可行点。分类性能提升不能掩盖这一事实。

### 8.2 Run26 只有一个 seed

Run18/21、Run22/23、Run24/25 已证明随机种子波动可大于单次注意力增益。Run26 必须补 seed42 和 seed43 才能判断空间门是否稳定。

### 8.3 partial-label 任务天然不平衡

训练集中分割监督、分类监督和双监督数量差异大；只有 736 个 train 样本同时有两类真值。模型可能在共享骨干中出现任务竞争，分类峰值、分割峰值和风险最优点也经常不在同一 epoch。

### 8.4 分类数据存在 group 元数据缺口

ISIC 2019 增量中 1,256 个样本缺失 lesion ID，只能按 singleton 处理。它们并不能证明 patient/lesion 独立，必须作为数据限制披露。

### 8.5 外部验证尚未执行

PAD-UFES-20 是手机临床图像，不是同域皮肤镜数据；只有六类支持且 MEL 仅 52 张。ISIC 2018 增量只有 89 对并缺 patient/lesion 元数据。两者都只能作为外部候选压力测试。

### 8.6 尚无最终盲测结论

IMA++ test 634 对仍被冻结，这是正确状态。项目目前所有 Run10~Run26 结论均来自 development val，不能被写成最终泛化性能。

### 8.7 校准、置信区间和效率报告未形成最终版

虽然训练 CSV 已输出 NLL/Brier/ECE，但最终候选仍需统一 bootstrap CI、逐类指标、参数/GFLOPs、推理速度、显存和校准分析。温度或阈值不能在 test/external 上拟合。

### 8.8 文件历史较长且存在易混淆项

- `experiments\17_YOLO26m_Binary8Class_BinaryLesionBoundaryGuidedAttention` 是未训练、已 superseded 的草案；
- 正式 Run17 是 `experiments\17_YOLO26m_Binary8Class_PartialLabel_Baseline`；
- 根目录旧 `yolo26n.pt` 与 `tools\yolo26n.pt` 容易混淆，Run18+ 配置锁定使用后者做 AMP 自检；
- `md` 目录中早期 handoff 是历史快照，不能覆盖本文件的 2026-09-05 状态；
- 根目录 `123`、缓存目录和旧草案不得由下一位 AI 在没有审计和用户授权时清理。

---

## 9. 项目当前正在执行的计划

当前主线是“在严格控制变量下验证注意力模块”，而不是无方向地继续堆模型：

1. Run18 建立修正生命周期的 YOLO26m plain seed42 基线；
2. Run21 加入 ECA-only seed42；
3. Run22/23 完成 seed43 plain/ECA 配对；
4. Run24/25 完成 seed44 plain/ECA 配对；
5. 三 seed 证明 ECA-only 无稳定净增益；
6. Run26 在 seed44 ECA 后单独增加恒等初始化空间门，得到有前景结果；
7. 当前暂停，等待下一位 AI 接手；
8. 下一步应优先补相同空间门的 seed42/43 配对，而不是启动未经预注册的新结构。

---

## 10. 推荐的未来计划

### 阶段 A：空间注意力多 seed 复现

推荐但尚未创建任何文件或启动训练：

1. **Run27 候选**：以 Run21（seed42 ECA-only）为基线，只增加与 Run26 完全相同的 P3/P4 spatial7 门；其余全部锁定。
2. **Run28 候选**：以 Run23（seed43 ECA-only）为基线，只增加相同 spatial7 门；其余全部锁定。
3. 每个新 run 均重新生成自己的 deterministic development projection、配置 hash、行为指纹、发布门和 checkpoint 路径；不得复制 Run26 的 release PASS 当授权。
4. 汇总三组 paired difference：Spatial+ECA − ECA-only，并同时报告相对 plain 的总效果。
5. 预先定义“稳定收益”的判断：至少 Joint 和 Group Macro-F1 的配对均值为正，方向大体一致，MEL→NV 不系统恶化；不要训练后再改判定规则。

### 阶段 B：决定注意力主线

- 若 seed42/43 复现支持 Run26：锁定 P3/P4 ECA+spatial 为论文注意力候选，进入统一多 seed 汇总与统计分析。
- 若复现方向不一致或均值接近 0：停止这一注意力架构方向，如实把它作为负向消融；不要继续叠加 P5、预测 mask 或更多门来追逐单次 val。

### 阶段 C：单独处理风险目标

只有在注意力结构结论完成后，才另开新编号实验处理 `mel→nv`：

- 从已锁定结构或最强 plain 基线出发；
- 每次只改变一个风险 loss 或决策因素；
- 不重复 Run11 的全 MEL 权重、Run14 的稀释 logit margin 或 Run15 的无效 pairwise 方案，除非提出明确的新机制；
- 可以考虑预注册 class-conditional focal/cost-sensitive objective、层级 malignant-vs-benign 辅助头或 val 内预先定义的代价矩阵，但不得同时改结构和 loss；
- 风险改进必须在效用门内评估，拒绝冷启动“假可行点”。

### 阶段 D：development-only 最终候选比较

在 test 解封前完成：

- 三 seed 汇总和配对统计；
- group/lesion bootstrap 95% CI；
- 八类逐类指标与混淆矩阵；
- NLL、Brier、ECE 和只用 development 数据的校准方案；
- 参数量、GFLOPs、RTX 5080 latency/throughput/显存；
- 定性分割、注意力热图与 hard cases；
- 写出候选 checkpoint SHA-256、阈值、温度和评价脚本 SHA-256；
- 创建正式模型锁文档。

### 阶段 E：最终盲测与外部评价

只有模型结构、checkpoint、阈值、温度和评价脚本全部冻结，并取得用户当次明确授权后：

1. 一次性运行 IMA++ internal blind test；
2. 单独运行 ISIC 2018 89 对 external segmentation candidate；
3. 单独运行 PAD-UFES-20 external classification candidate；
4. PAD 同时报告 full-8 macro 与 supported-6 macro；
5. 不根据 test/external 结果回训或换 checkpoint；
6. 如产生新假设，必须作为新的 exploratory run 重新预注册，不能污染主结果。

### 阶段 F：论文整理

- 画出旧八通道任务到新双头 partial-label 任务的演进图；
- 建立 Run10~Run28 的单因素谱系图；
- 主表使用同一指标口径和多 seed 统计；
- 把 Run19/20、ECA-only 等负结果如实作为消融；
- 分开报告 image 与 group 指标；
- 说明模型不是临床诊断产品；
- 披露公开数据域、类别不平衡、缺 patient/lesion ID、外部模态差异和风险门未完全解决等限制。

---

## 11. 当前文件结构

```text
PiFuV2/
├─ .venv/                         PyCharm 项目解释器
├─ Data/
│  ├─ 1/                          IMA++、ISIC 2018 等官方源文件，只读保留
│  ├─ external_candidates/        PAD-UFES-20 官方外部候选源文件
│  ├─ processed_skin7_v3/         Run1~Run9 历史数据，只作复现
│  ├─ processed_imaplusplus_binary_v1/
│  │                               新二值分割数据，train/val/test=10483/292/634
│  ├─ processed_isic2019_classification_incremental_v1/
│  │                               新八分类增量数据，train/val=12979/1452
│  ├─ processed_isic2018_task1_incremental_v1/
│  │                               89 对 external segmentation candidate
│  └─ processed_pad_ufes_20_external_classification_v1/
│                                  2269 张 external classification candidate
├─ experiments/
│  ├─ 01_... 至 09_...            旧 V3 历史实验
│  ├─ 10_... 至 16_...            新双任务体系早期实验
│  ├─ 17_... 至 25_...            YOLO26m 生命周期与注意力消融
│  └─ 26_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed44/
│                                  最新已完成正式实验
├─ md/                             历史 handoff、数据验收、Run10~Run17 文档
├─ models/
│  ├─ yolo26s-sem.pt               Run10~Run15 正式预训练权重
│  └─ yolo26m-sem.pt               Run16~Run26 正式预训练权重
├─ preflight/                      被替代草案与预检归档
├─ tests/                          公共测试
├─ tools/
│  ├─ run1/ ... run26/             各轮 Python/YAML/协议/发布/预检文件
│  ├─ classification_taxonomy_v1.yaml
│  ├─ imaplusplus_partial_multitask_v1.py
│  ├─ multitask_checkpoint_contract.py
│  ├─ evaluate_imaplusplus_multitask_v1.py
│  └─ yolo26n.pt                   Run18+ 本地 AMP 自检模型
├─ yolo26n.pt                      历史根目录轻量模型，勿与正式主干混淆
└─ PROJECT_HANDOFF_2026-09-05.md   本交接文档，当前最新状态
```

未来所有 Run27 文件必须放入 `tools\run27`，Run28 放入 `tools\run28`，以此类推；正式结果写入新的 `experiments\27_...`、`experiments\28_...`，不得回写旧目录。

---

## 12. Run26 关键路径

工具与协议：

- `tools\run26\train_run26_from_pycharm.py`
- `tools\run26\run26_preregistered_config_v1.yaml`
- `tools\run26\RUN26_PROTOCOL_2026-09-02.md`
- `tools\run26\PROJECT_RUN26_RELEASE_2026-09-02.md`
- `tools\run26\RUN26_PREFLIGHT_VERIFICATION_2026-09-02.md`
- `tools\run26\imaplusplus_partial_multitask_p3p4_eca_spatial_attention_seed44_v1.py`

正式结果目录：

`experiments\26_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed44\results\run26_yolo26m_p3p4_eca_spatial_attention_seed44`

关键文件：

- `RUN26_TRAINING_COMPLETE.json`
- `RUN26_LAUNCH_RECORD.json`
- `results.csv`
- `results.png`
- `weights\best.pt`
- `weights\last.pt`
- `checkpoint_bundles\aliases\best_joint_score.json`
- `checkpoint_bundles\aliases\best_binary_iou.json`
- `checkpoint_bundles\epoch_0040\trainer.pt`
- `checkpoint_bundles\epoch_0040\runtime_state.pt`

下一位 AI 不应继续 Run26；应把 `epoch_0040` 当 Run26 的 development-val 最佳 joint 诊断点，并在正式锁定前保持 test/external 封闭。

---

## 13. 下一位 AI 的强制执行纪律

1. 先完整阅读本文件、`md\DATA_TASK_ACCEPTANCE_2026-08-04.md`、`md\CLASSIFICATION_MULTITASK_CONTRACT_2026-08-04.md`、Run26 protocol/config 和目标新 run 的前一轮 decision 文档。
2. 不改新数据 manifest、split、mask 值域、taxonomy、test 或 external 角色。
3. 不覆盖 Run1~Run26；不要删除未确认的旧草案、缓存、模型或备份。
4. 每次只改变一个预注册因素，并明确指定基线 run、seed 和差值口径。
5. 新 run 先创建 `tools\runXX`，完成 py_compile、静态测试、真实 development batch、GPU/AMP 容量、CPU/GPU resume 等价性、prepare-only 和正式 Trainer 构造。
6. 发布文档必须先保持 `BLOCKED_PENDING_USER_CONFIRMATION`；只有用户明确确认后才能改为 PASS。
7. 正式训练必须从 PyCharm 的锁定入口启动，默认 `resume=auto`；不要用临时控制台命令绕过发布门。
8. 中断后只恢复最新完整 bundle；若 epoch 1 完成前没有完整 bundle，应归档半成品并按协议 fresh start。
9. 正常早停/完成后不得继续同名实验。
10. bundle epoch 是零基 completed epoch，CSV epoch 是一基可见 epoch，比较时必须 `csv_epoch=bundle_epoch+1`。
11. 评价时区分独立峰值与同一 checkpoint 指标，不能把不同 epoch 的各列峰值拼成一个不存在的模型。
12. `best_macro_f1` alias 实际跟踪 group Macro-F1；原始 image Macro-F1 峰值应直接从 CSV 读取。
13. formal feasible 必须同时满足效用门与风险门；冷启动低风险点不算候选。
14. 正式模型锁前禁止 test/external；解封仍需用户当次明确授权。
15. 不把 smoke、prepare-only、Trainer 构造或训练日志峰值写成最终论文 test 结果。

---

## 14. 可直接交给下一位 AI 的开场提示

```text
请先完整阅读项目根目录 E:\App\DaiMa\JetBrains\Project\PiFuV2\PROJECT_HANDOFF_2026-09-05.md，并继续阅读 md\DATA_TASK_ACCEPTANCE_2026-08-04.md、md\CLASSIFICATION_MULTITASK_CONTRACT_2026-08-04.md 与 tools\run26\RUN26_PROTOCOL_2026-09-02.md。

Run1~Run26 已完成或按历史状态封存，全部必须保留。Run26 已正常早停完成：CSV epoch 1~71 连续，最佳 joint bundle 为零基 epoch_0040（可见 epoch 41），Binary IoU=0.76971、Image Macro-F1=0.55333、Group Macro-F1=0.63762、Group Recall=0.65132、MEL→NV=0.33673、Joint=0.70056。5 个完整 checkpoint bundle 的大小、SHA-256、_SUCCESS 和 fingerprint 全部复核通过；没有使用 test 或 external。

Run26 相对同 seed44 Run25 ECA-only 的 Joint +0.01007、Group Macro-F1 +0.01391、Image Macro-F1 +0.02572、IoU +0.00530、MEL→NV -0.11225，是有前景的单 seed 结果；但正式 feasible checkpoint 仍为 0，不能锁为最终模型。

当前不启动下一轮。建议下一阶段先预注册 seed42 和 seed43 的相同 P3/P4 ECA+spatial7 配对复现，每次仅增加与 Run26 完全相同的空间门。新文件放入 tools\run27、tools\run28，新结果写入独立 experiments 编号目录。完成三 seed 后再决定是否锁定注意力主线或转向单独的风险损失实验。

不得改动冻结数据 manifest/split/mask/taxonomy，不得运行 IMA++ test、ISIC2018 external 或 PAD-UFES-20。每个正式 run 必须完成预注册、静态/GPU/恢复预检、发布门和用户明确确认，并从 PyCharm 锁定入口启动。
```

---

## 15. 最终判断

项目已经从早期“七病种标签广播到像素”的任务定义，演进为更合理的“二值病灶分割 + 八类图像分类”partial-label 多任务体系；分类 taxonomy、官方数据去重、loader、模型、统一 val 指标、完整恢复和外部数据角色等原始阻塞问题均已解决。

Run18~Run25 已建立较严格的 YOLO26m 生命周期和三 seed ECA-only 消融，证明纯 ECA 不是稳定改进。Run26 新增的恒等初始化 P3/P4 空间门在 seed44 上获得了当前阶段最有希望的联合改善，是一次成功训练，但证据强度仍不足以结束消融，也未解决临床风险门。

因此最合理的后续顺序是：先补 Run26 结构的 seed42/43 配对复现；再决定注意力主线；随后以独立单因素实验处理 MEL→NV 风险；最后完成多 seed 统计、统一 val 锁模、效率与校准、用户授权下的一次盲测和外部压力测试。任何提前运行 test、根据 external 回训、继续已完成 Run26，或同时改变结构与损失的做法，都会破坏论文的因果解释和结果可信度。
