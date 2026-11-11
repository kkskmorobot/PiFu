# PiFuV2 项目超详细交接、runs4 训练总结与当前目录状态

> 更新时间：2026-07-20（完成备份、目录整理与清理后）  
> 项目路径：`E:\App\DaiMa\JetBrains\Project\PiFuV2`  
> 论文题目：**《基于注意力机制改进 YOLO 的皮肤病识别算法研究》**  
> 当前状态：runs4 已在第 91 轮正常触发早停；训练后审计、项目备份、目录整理和旧文件清理均已完成；当前没有训练进程，下一步是修正统一评价器并设计第五次实验。  
> 重要纪律：不要删除或覆盖 `experiments\01_...` 至 `experiments\04_...`；从备份恢复时只能复制、不能剪切；在最终模型和评价协议确定前，不要继续读取 test 指标。

---

## 0. 给新对话的最短结论

PiFuV2 已经完成了数据清洗、按 lesion_id 防泄漏划分、训练集 NV 限制、少数类数据增强、YOLO11s 实例分割基线，以及 YOLO26s 语义分割注意力基线。当前最稳定的数据版本是 `Data/processed_skin7_v3/data.yaml`。

runs4 使用 `yolo26s-sem.pt`，在第 91 轮因 30 轮没有超过第 61 轮的 mIoU 而早停。第 61 轮是语义 mIoU 最优点：**mIoU=0.59122，pixel accuracy=0.92500**。修正独立验证的 background 类别映射后，七种疾病的前景平均 IoU 约为 **0.5388**。图像级诊断结果为：**Accuracy=0.8092、Macro-P=0.7889、Macro-R=0.6833、Macro-F1=0.7041**。

runs4 不是全面胜过 runs3：它明显提高了 bcc 和 vasc，并把 `mel→nv` 从 26 张降到 18 张，但 df 召回率从 55.56% 降到 33.33%，`nv→mel` 从 66 张增到 84 张，整体准确率也从 82.38% 降到 80.92%。因此 runs4 应被视为一个有价值的 **YOLO26 语义分割/原生注意力基线**，不能直接宣布为最终模型。

下一步不应继续盲目调数据或单纯加大训练轮数。最合理的路线是：先修正统一评价器和模型选择指标，再进行严格的注意力消融。建议实验顺序为：

1. runs5：YOLO26-sem + 辅助图像级分类头，不加入新注意力，用于验证“显式诊断头”本身的作用；
2. runs6：runs5 + 标准 CBAM，作为通用注意力对照；
3. runs7：runs5 + 病灶掩码引导的多尺度双注意力模块，作为论文提出方法；
4. 筛选阶段统一 seed=42；最终候选再运行至少 3 个随机种子，并报告均值、标准差和 lesion 级置信区间。

### 0.1 目录整理后的最新状态

- 项目入口：`PROJECT_INDEX.md`；
- 当前唯一数据集：`Data\processed_skin7_v3`；
- 当前唯一有效环境：`.venv`；
- 公共脚本：`tools`；
- 预训练模型：`models`；
- 历史与清理文档：`docs`；
- 四次正式实验：`experiments\01_YOLO11s_V1_Baseline` 至 `experiments\04_YOLO26s_Semantic_C2PSA`；
- 非数据、非虚拟环境备份：`E:\App\DaiMa\JetBrains\Project\BeiFen\PiFuV2`；
- 备份包含 268 个文件、2,310,555,920 字节，逐文件 SHA-256 校验为 0 缺失、0 不一致；
- 已清理旧数据、无效环境和缓存约 10.489 GiB；
- 四次正式实验与备份逐文件复核结果：第一次训练 34/34、第一次 test 16/16、第二次训练 31/31、第二次 test 16/16、第三次训练 32/32、第四次训练与分析 81/81，全部 0 缺失、0 哈希不一致；
- 工作目录中的 runs2/runs3 额外诊断资料已采用“复制”方式加入第二、第三次实验的 `analysis`，原始源文件仍保留。

---

## 1. 整体任务目标

### 1.1 工程目标

构建一个能够对皮肤镜图像完成以下任务的模型：

- 识别 7 类皮肤病；
- 给出病灶的像素级分割区域；
- 在类别不平衡情况下不过度偏向 NV；
- 重点提升 akiec、bcc、df 等少数类的召回率；
- 重点减少恶性黑色素瘤 mel 被误判为 nv 的风险；
- 保持训练可恢复、实验结果可复现、历史实验不覆盖。

### 1.2 论文目标

这个项目不是单纯“把准确率训练高”，而是一个算法研究型毕业设计。论文要回答的核心问题是：

1. 原始 YOLO 基线在 HAM10000 七分类皮肤病分割/识别上有哪些瓶颈？
2. 数据不平衡、病灶重复拍摄、诊断来源和设备域偏差对结果有什么影响？
3. 注意力机制能否让模型更关注病灶内部的颜色、纹理和边界，而不是背景或数据来源特征？
4. 提出的注意力改进是否在公平控制变量的条件下，稳定提高 Macro-F1、少数类召回率和前景分割质量？
5. 提升是否来自注意力模块本身，而不是数据、损失函数、训练轮数或随机种子变化？

### 1.3 建议的最终成功标准

最终模型不应只用 Accuracy 或 pixel accuracy 判断。建议至少同时满足：

- 图像级 Macro-F1 高于 runs3 和 runs4；
- akiec、bcc、df 的召回率有稳定提升；
- `mel→nv` 数量不高于 runs4，最好继续下降；
- 前景七病种 mean IoU 不低于 runs4 的约 0.5388；
- 多随机种子结果不是偶然提升；
- 参数量、GFLOPs 和推理速度可接受；
- test 只在方法最终固定后使用。

---

## 2. 项目类型以及为什么这样定义

### 2.1 当前项目的技术类型

当前项目属于：

- **医学图像分析**；
- **七类别皮肤病识别**；
- **像素级语义/实例分割**；
- **长尾类别学习**；
- **基于注意力机制的 YOLO 改进研究**；
- **多任务学习候选项目**，因为后续最好同时优化分割与图像级诊断。

### 2.2 为什么不是纯分类项目

数据中不仅有图像级疾病类别，还有病灶掩码。论文标题又明确是改进 YOLO，因此只训练普通分类网络会丢失“病灶在哪里”的监督，也无法研究空间注意力和边界特征。

### 2.3 为什么 runs3 是实例分割、runs4 是语义分割

- runs3 的 YOLO11s-seg 为每个病灶输出实例框、类别和实例 mask，主要指标是 Mask mAP。
- runs4 的 YOLO26s-sem 为每个像素直接输出 0~6 疾病类别或 7=background，主要指标是 mIoU。
- HAM10000 图像通常只有一个居中的主要病灶，因此语义分割是合理的，但疾病识别不能只靠“哪类像素面积最大”来完成。
- 两种任务的 mAP 与 mIoU 不能直接横向比较，只能用统一的图像级 Precision/Recall/F1 做诊断能力比较。

### 2.4 为什么它不是临床诊断产品

当前模型只在单一公开数据集及内部划分上开发，没有完成外部数据验证、医院域验证、校准、公平性分析和临床风险评估。论文中应称为“辅助识别算法研究”，不能宣称可直接用于临床诊断。

---

## 3. 类别定义

固定类别顺序如下，任何 YAML、标签、模型输出和混淆矩阵都必须保持一致：

| ID | 缩写 | 含义 |
|---:|---|---|
| 0 | akiec | 光化性角化病/鲍恩病相关病变 |
| 1 | bcc | 基底细胞癌 |
| 2 | bkl | 良性角化性病变 |
| 3 | df | 皮肤纤维瘤 |
| 4 | mel | 黑色素瘤 |
| 5 | nv | 黑色素细胞痣 |
| 6 | vasc | 血管性病变 |
| 7 | background | 仅用于 YOLO26 语义分割的背景像素 |

---

## 4. 环境与硬件

- 系统：Windows；
- IDE：PyCharm；
- PyCharm：`E:\App\DaiMa\JetBrains\PyCharm\bin\pycharm64.exe`；
- 项目解释器：`E:\App\DaiMa\JetBrains\Project\PiFuV2\.venv\Scripts\python.exe`；
- Python：3.13.7；
- PyTorch：2.8.0+cu129；
- Ultralytics：8.4.101；
- PyYAML：6.0.2；
- GPU：NVIDIA GeForce RTX 5080，16 GB 显存；
- runs4 在 640 分辨率、batch=16 下峰值显存约 5.67 GB。

当前只保留有效的 `.venv`。历史目录 `.venv_broken_python313`、`.venv_python312_empty` 和 `.venv_yolo26` 已在确认后删除。`.venv` 约 7.965 GiB，主要体积来自 PyTorch、CUDA 运行库和训练依赖，属于当前训练环境的正常占用。

---

## 5. 原始数据问题与已完成的清理

### 5.1 最初项目为什么重做

最初 PiFu 项目出现了明显过拟合和 NV 偏置。若直接使用原始类别分布，NV 数量过多，模型容易把大量图片预测成 NV。后来决定不在旧工程上继续修补，而是在 PiFuV2 中重新建立可审计的数据流水线。

### 5.2 官方 Task 1/Task 3 数据问题

- 官方分割任务和疾病分类任务并不是所有测试样本都能同时获得病种标签与分割掩码。
- 用户下载了两个 Task 1 相关文件，但其中一些图片仍无法确定七类疾病标签，因此不能强行加入监督训练。
- 有一张无法在用户保存的官方原始备份中找到、也无法确认病种的孤立样本，已按用户决定从工作数据中删除；原始备份仍由用户保留。
- 少量异常掩码暂时保留并记录，未自动删除。

### 5.3 当前为什么使用 HAM10000 内部分割

当前内部 train/val/test 都来自同时具有疾病标签和病灶 mask 的 HAM10000 数据。官方 ISIC Task 1/Task 3 测试下载没有直接混入当前内部测试集。

### 5.4 当前磁盘中的数据状态

项目内现在只保留 `Data\processed_skin7_v3`，它完整包含 images、labels、metadata、data.yaml、summary.json 以及 8512/1504/1503 的 train/val/test。原始 `Data\train`、`Data\test`、V1 和 V2 处理数据已按用户确认删除；这些旧数据未进入本次非数据备份，如需重建必须使用用户另行保存的官方原始备份。

---

## 6. 数据版本演变

### 6.1 V1：`Data/processed_skin7`

> 历史数据版本说明：V1 数据目录已从当前项目清理；第一次正式训练结果、配置和报告仍完整保存在 `experiments\01_YOLO11s_V1_Baseline`。

- 按 `lesion_id` 划分，防止同一病灶的不同图片跨集合；
- NV 训练图片限制在约 3500；
- 对少数类做离线增强；
- 作为 `runs` 的 YOLO11s-seg 初始基线。

### 6.2 V2：`Data/processed_skin7_v2`

> 历史数据版本说明：V2 数据目录已从当前项目清理；第二次正式训练、早期 test 结果、配置、报告和诊断图表仍完整保存在 `experiments\02_YOLO11s_V2_DataBalance`。

- 训练分布：akiec1000、bcc1200、bkl1400、df400、mel1800、nv2800、vasc450；
- 增加了更广泛的平衡、hard-example 优先和旋转/缩放/平移策略；
- `runs2` 的结果反而下降，Mask mAP50-95 的训练日志峰值约从 V1 的 0.5801 降到 0.5306；
- 说明“更均衡、更多增强”不等于“更好”，重复增强可能放大少数类有限原始病灶的记忆。

### 6.3 V3：`Data/processed_skin7_v3`

V3 是当前固定的稳定数据版本。它从 V1 出发，只把训练 NV 从 3500 减到 3200，其他非 NV 训练数据、增强逻辑、val 和 test 保持不变。

| Split | akiec | bcc | bkl | df | mel | nv | vasc | 总计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 800 | 1000 | 1400 | 320 | 1400 | 3200 | 392 | 8512 |
| val | 49 | 77 | 165 | 18 | 167 | 1006 | 22 | 1504 |
| test | 49 | 77 | 165 | 17 | 167 | 1006 | 22 | 1503 |

V3 已验证：

- train/val/test 图片与标签一一对应；
- 标签非空；
- 类别 ID 全部合法；
- 多边形至少 3 个点，坐标数量为偶数；
- 归一化坐标位于 `[0,1]`；
- lesion_id 跨集合重叠均为 0；
- train/val/test 的独立 lesion 数分别为 4042、1118、1120。

### 6.4 V3 的真实独立病灶与合成比例

| 类别 | train 图片 | 独立 lesion | 离线增强图片 | 合成比例 |
|---|---:|---:|---:|---:|
| akiec | 800 | 160 | 571 | 71.4% |
| bcc | 1000 | 229 | 640 | 64.0% |
| bkl | 1400 | 510 | 631 | 45.1% |
| df | 320 | 51 | 240 | 75.0% |
| mel | 1400 | 431 | 621 | 44.4% |
| nv | 3200 | 2593 | 0 | 0% |
| vasc | 392 | 68 | 294 | 75.0% |

关键解释：图片数量看上去已经平衡，但独立病灶远没有平衡。df 只有 51 个、vasc 只有 68 个、akiec 只有 160 个独立训练病灶。对同一病灶生成很多变体不能替代新的临床病例。

---

## 7. 当前数据的优点

1. 已按 lesion_id 划分，不存在同一病灶跨 train/val/test 的直接泄漏。
2. val 与 test 没有做离线增强。
3. V3 相对 V1 是严格单变量 NV cap 消融，便于论文解释。
4. 每个增强样本能追溯到 `source_image_id`。
5. 掩码与图像同步变换，不会出现图像增强后 mask 未同步的问题。
6. 已经避免原始 NV 数量极端压制少数类。
7. 数据构建脚本拒绝静默覆盖已有输出，有利于复现和保留实验。

---

## 8. 当前数据仍然存在的缺点

### 8.1 真正的少数类仍然很少

df、vasc、akiec 的独立病灶非常少。runs4 的 df 验证集只有 18 张、11 个独立 lesion；vasc 只有 22 张、15 个 lesion。一次预测变化就会让召回率大幅波动。

例如 runs4 的图像级召回率 95% Wilson 区间大致为：

| 类别 | Recall | 约 95% 区间 |
|---|---:|---:|
| akiec | 0.449 | 0.319–0.587 |
| bcc | 0.818 | 0.718–0.888 |
| bkl | 0.782 | 0.713–0.838 |
| df | 0.333 | 0.163–0.563 |
| mel | 0.784 | 0.716–0.840 |
| nv | 0.844 | 0.820–0.865 |
| vasc | 0.773 | 0.566–0.899 |

这说明 df/vasc 单次划分的数字不稳定，必须增加多随机种子或 lesion 级交叉验证。

### 8.2 离线增强比例过高

akiec、df、vasc 等类别的 70%~75% 训练图片是同一批原始病灶的变体。模型可能学会某些病灶的纹理模板，而不是泛化到新病例。

### 8.3 数据来源与类别强相关

原始 HAM10000 的诊断确认方式与类别分布并不一致：

- akiec、bcc、mel 基本以 histology 为主；
- bkl 同时包含 histology、consensus 和 confocal；
- df、vasc 有相当一部分来自 consensus；
- nv 大量来自 follow_up，也包含 histology 和 consensus。

采集数据源也与类别强相关，例如 akiec 主要来自 rosendahl，nv 大量来自 vidir_molemax，而 mel 更多来自 vidir_modern。模型可能学习设备、色彩、裁剪和背景域特征，而不是疾病本身。

### 8.4 bkl 类别内部异质性高

bkl 是多种良性角化病变的集合，内部外观跨度很大，很容易与 akiec、mel、nv 混淆。

### 8.5 当前 test 的方法学风险

runs4 本次没有读取 test，当前新增审计也只使用 val。但是 `DATA_V3_REPORT.md` 明确写到 V3 是在观察到 V2 的 held-out test performance 下降后建立的。因此从整个项目历史的严格统计意义看，现有 test 已经被间接用于开发决策，不能再完全称为“从未看过的终极盲测集”。

建议论文采用以下补救方案之一：

1. 将当前 test 明确称为内部开发留出集，并对最终模型增加真正外部数据验证；
2. 在 HAM10000 上做 lesion 级多折交叉验证，报告所有折的均值和置信区间；
3. 若仍使用当前 test，必须在论文中如实说明早期实验曾观察过其总体结果，避免夸大独立性。

---

## 9. 已完成实验谱系

| 实验 | 模型/任务 | 数据 | 目的 | 结论 |
|---|---|---|---|---|
| 第一次（原 runs） | YOLO11s-seg 实例分割 | V1 | 初始稳定基线 | Mask mAP50-95 日志峰值约 0.5801 |
| 第二次（原 runs2） | YOLO11s-seg 实例分割 | V2 | 更强平衡与 hard mining | 第 58 轮早停，Mask mAP50-95 日志峰值约 0.5306，明显下降 |
| 第三次（原 runs3） | YOLO11s-seg 实例分割 | V3 | 仅把 NV 限制到 3200 | 恢复到接近 V1，成为稳定实例分割基线 |
| 第四次（原 runs4） | YOLO26s-sem 语义分割 | V3 | 换用 YOLO26、语义头、原生 C2PSA 注意力和温和增强 | 第 91 轮早停；部分少数类提升，但错误结构仍明显 |

历史实验结果现在分别保存在 `experiments\01_YOLO11s_V1_Baseline`、`02_YOLO11s_V2_DataBalance`、`03_YOLO11s_V3_NV3200` 和 `04_YOLO26s_Semantic_C2PSA`。后续从 `experiments\05_...` 继续编号。

---

## 10. runs3 稳定基线

模型：YOLO11s-seg，约 10.07M 参数、32.8 GFLOPs。

整体实例分割验证结果：

- Box P=0.7117；
- Box R=0.6784；
- Box mAP50=0.7444；
- Box mAP50-95=0.5957；
- Mask P=0.7112；
- Mask R=0.6755；
- Mask mAP50=0.7421；
- Mask mAP50-95=0.5752。

图像级诊断结果（val 1504 张，confidence=0.25）：

- detection coverage=0.990；
- Top-1 Accuracy=0.8238；
- Macro-P≈0.7278；
- Macro-R≈0.6843；
- Macro-F1≈0.6994。

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| akiec | 0.6316 | 0.4898 | 0.5517 |
| bcc | 0.7297 | 0.7013 | 0.7152 |
| bkl | 0.6648 | 0.7212 | 0.6919 |
| df | 0.7692 | 0.5556 | 0.6452 |
| mel | 0.5650 | 0.7545 | 0.6461 |
| nv | 0.9449 | 0.8857 | 0.9143 |
| vasc | 0.7895 | 0.6818 | 0.7317 |

runs3 最大错误方向：

- nv→mel：66；
- nv→bkl：34；
- mel→nv：26；
- bkl→mel：17；
- bkl→nv：12；
- mel→bkl：11；
- nv→bcc：9；
- akiec→mel：8；
- akiec→bkl：8；
- bcc→nv：8。

正确类别样本的平均 Mask IoU 大约 0.776~0.926，且低于 0.5 的样本很少。这说明 runs3 的首要瓶颈是疾病分类，不是病灶定位。

---

## 11. runs4 的训练设计

主要文件：

- `tools\train_skin7_yolo26_sem.py`；
- `experiments\04_YOLO26s_Semantic_C2PSA\train_yolo26s_sem_v4.yaml`；
- `tools\resume_yolo26_sem_v4.py`；
- `tools\evaluate_yolo26_sem_image_level.py`。

配置：

- 模型：`yolo26s-sem.pt`；
- 任务：semantic segmentation；
- 参数量：6,490,144；
- GFLOPs：17.1；
- 相比 runs3，参数量约减少 35.5%，GFLOPs 约减少 47.9%；
- 原生注意力：C2PSA；
- epochs=150；
- patience=30；
- batch=16；
- imgsz=640；
- optimizer=AdamW；
- lr0=0.001；
- cosine learning rate；
- seed=42；
- deterministic=true；
- `cls_pw=0.5`，使用较保守的像素频率类别权重；
- 类别权重大约为 `[1.04, 1.04, 0.92, 1.401, 0.933, 0.76, 1.51, 0.332]`，分别对应七病种与 background；
- Mosaic、MixUp、CutMix、Copy-Paste 全部关闭；
- 仅保留温和 HSV、±5°旋转、3%平移、10%缩放和翻转。

设计目的：减少 runs3 中过强 Mosaic/HSV/scale 与离线增强叠加造成的临床颜色、纹理和边缘破坏，并降低 NV/background 对损失的主导。

---

## 12. runs4 的早停与过拟合分析

### 12.1 早停是否正常

正常。第 61 轮达到最高 mIoU=0.59122。之后连续 30 轮没有超过它，因此第 91 轮触发 patience=30 的早停。

第 80 轮曾回升到 mIoU=0.58921，但仍低于第 61 轮，因此不会重置早停计数。

### 12.2 训练曲线关键点

| Epoch | mIoU | pixel acc | train CE | train Dice | val CE | val Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.30386 | 0.83809 | 1.00474 | 0.65009 | 0.69374 | 0.70563 |
| 10 | 0.45851 | 0.88312 | 0.48407 | 0.42224 | 0.49913 | 0.62700 |
| 20 | 0.47156 | 0.88711 | 0.32821 | 0.34861 | 0.46856 | 0.60727 |
| 30 | 0.54294 | 0.91115 | 0.25305 | 0.26236 | 0.48714 | 0.49615 |
| 40 | 0.54405 | 0.91927 | 0.18451 | 0.16365 | 0.58634 | 0.37655 |
| 50 | 0.57763 | 0.92505 | 0.14767 | 0.14212 | 0.58245 | 0.38158 |
| 61 | **0.59122** | 0.92500 | 0.10794 | 0.11819 | 0.71790 | 0.35952 |
| 70 | 0.54671 | 0.92022 | 0.09288 | 0.06865 | 0.92959 | 0.28729 |
| 80 | 0.58921 | 0.92600 | 0.07670 | 0.05845 | 0.89391 | 0.27167 |
| 91 | 0.57104 | 0.92590 | 0.06481 | 0.04274 | 1.07085 | 0.27347 |

### 12.3 过拟合结论

- val CE 在第 24 轮达到最低 0.45079，之后总体上升；
- train CE 一直下降到第 91 轮的 0.06481；
- 从第 61 轮到第 91 轮，train CE 继续下降约 40%，但 val CE 上升约 49.2%；
- mIoU 从 0.59122 降到 0.57104；
- val Dice 仍继续改善，说明轮廓覆盖能力在提高，但多类别置信度与类别区分变差；
- 这属于明显的分类/置信度过拟合，而不是训练崩溃。

结论：增加 epochs 或关闭早停没有价值，反而更可能强化少数原始病灶和数据源偏差。

---

## 13. 独立语义验证发现的 background 映射问题

### 13.1 问题表现

直接使用原始 7 类 `data.yaml` 对保存后的 `best.pt` 调用 standalone `model.val()` 时，Ultralytics 8.4.101 的验证数据对象仍为 7 类，`PolygonSemanticDataset` 把 `nc-1=6` 当成 background，导致背景像素被合并进 vasc。

错误验证会产生：

- mIoU 约 0.431；
- pixel accuracy 约 0.202；
- vasc 像素支持异常巨大；
- background 目标像素为 0。

这些数字是验证器类别映射错误，不代表模型真实性能。

### 13.2 已确认的修正

独立验证时需要显式声明：

- `nc: 8`；
- `bg_class_idx: 7`；
- `_polygon_bg_added: true`；
- names 中加入 `7: background`。

修正后重新验证得到与训练日志一致的 mIoU=0.591、pixel accuracy=0.925。

官方文档说明，多类别 YOLO polygon 语义数据应自动追加 background；当前安装版本的 standalone 验证实际行为与预期不一致。后续必须保留显式 8 类验证 YAML，不能直接用原始 7 类 YAML 做语义 val/test。

### 13.3 当前代码缺陷

`tools\train_skin7_yolo26_sem.py` 中 `evaluate_test()` 仍把原始 7 类 data YAML 传给 standalone `model.val()`。在最终 test 前必须先修复，否则 test 的 vasc/background 指标会错误。当前 test 尚未因这次 runs4 审计而被运行。

---

## 14. runs4 正确的语义分割结果

正确验证对象：`best.pt`，对应训练结果第 61 轮。Ultralytics 的周期文件名从 0 开始，因此 `epoch60.pt` 与第 61 轮训练状态对应，图像级结果与 `best.pt` 完全一致。

| 类别 | IoU | 类别像素准确率 |
|---|---:|---:|
| akiec | 0.356 | 0.394 |
| bcc | 0.537 | 0.698 |
| bkl | 0.476 | 0.770 |
| df | 0.410 | 0.468 |
| mel | 0.474 | 0.746 |
| nv | 0.764 | 0.834 |
| vasc | 0.755 | 0.785 |
| background | 0.959 | 0.976 |

汇总：

- 8 类 mIoU=0.591；
- overall pixel accuracy=0.925；
- 仅七种疾病的前景 mean IoU≈0.5388；
- background 占全部验证像素约 74.10%；
- NV 占约 15.75%；
- background+NV 合计约 89.85%。

因此 92.5% pixel accuracy 很大程度受 background 和 NV 支配，不能作为主要论文结论。前景 mean IoU、逐类 IoU 和 Macro-F1 更重要。

---

## 15. runs4 图像级诊断结果

评价方式：排除 background，统计每张语义图中七疾病类别的像素面积；若疾病像素占比低于 1%，记为 no_prediction；否则取面积最大的疾病类别为图像级预测。

汇总：

- val images=1504；
- no_prediction=16；
- Top-1 Accuracy=0.80918；
- Macro-Precision=0.78891；
- Macro-Recall=0.68334；
- Macro-F1=0.70412。

| 类别 | Support | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 49 | 0.8148 | 0.4490 | 0.5789 | 22 | 5 | 27 |
| bcc | 77 | 0.7778 | 0.8182 | 0.7975 | 63 | 18 | 14 |
| bkl | 165 | 0.5785 | 0.7818 | 0.6649 | 129 | 94 | 36 |
| df | 18 | 0.8571 | 0.3333 | 0.4800 | 6 | 1 | 12 |
| mel | 167 | 0.5413 | 0.7844 | 0.6406 | 131 | 111 | 36 |
| nv | 1006 | 0.9529 | 0.8439 | 0.8951 | 849 | 42 | 157 |
| vasc | 22 | 1.0000 | 0.7727 | 0.8718 | 17 | 0 | 5 |

最大错误方向：

- nv→mel：84；
- nv→bkl：51；
- mel→nv：18；
- mel→bkl：16；
- akiec→bkl：15；
- bkl→mel：14；
- nv→no_prediction：13；
- bkl→nv：13；
- bcc→bkl：7；
- akiec→mel：7；
- nv→bcc：7；
- df→nv：4；
- df→bkl：4。

---

## 16. runs4 与 runs3 的公平对比

下面只比较相同 val 上的图像级诊断指标；Mask mAP 与语义 mIoU 不直接比较。

| 类别 | runs3 P | runs3 R | runs3 F1 | runs4 P | runs4 R | runs4 F1 | F1 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 0.6316 | 0.4898 | 0.5517 | 0.8148 | 0.4490 | 0.5789 | +0.0272 |
| bcc | 0.7297 | 0.7013 | 0.7152 | 0.7778 | 0.8182 | 0.7975 | +0.0823 |
| bkl | 0.6648 | 0.7212 | 0.6919 | 0.5785 | 0.7818 | 0.6649 | -0.0269 |
| df | 0.7692 | 0.5556 | 0.6452 | 0.8571 | 0.3333 | 0.4800 | -0.1652 |
| mel | 0.5650 | 0.7545 | 0.6461 | 0.5413 | 0.7844 | 0.6406 | -0.0056 |
| nv | 0.9449 | 0.8857 | 0.9143 | 0.9529 | 0.8439 | 0.8951 | -0.0192 |
| vasc | 0.7895 | 0.6818 | 0.7317 | 1.0000 | 0.7727 | 0.8718 | +0.1401 |

整体：

| 指标 | runs3 | runs4 | 结论 |
|---|---:|---:|---|
| Accuracy | 0.8238 | 0.8092 | runs4 下降 1.46 个百分点 |
| Macro-P | 0.7278 | 0.7889 | runs4 明显提高 |
| Macro-R | 0.6843 | 0.6833 | 基本不变 |
| Macro-F1 | 0.6994 | 0.7041 | runs4 轻微提高约 0.47 个百分点 |

与最初优化目标的关系：

- bcc 召回率：0.7013→0.8182，明显成功；
- akiec 召回率：0.4898→0.4490，未成功；
- df 召回率：0.5556→0.3333，明显失败；
- mel→nv：26→18，成功减少 8 张；
- nv→mel：66→84，反向错误恶化；
- vasc：显著改善，但验证样本很少，需要多 seed 确认。

最终判断：runs4 证明 YOLO26-sem 与原生 C2PSA 有潜力，并且更轻量，但没有解决整体长尾识别问题。

---

## 17. 周期快照的图像级评价

保存的每 10 轮快照已经全部在 val 上重新评价：

| 文件 | 对应训练状态 | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| epoch0.pt | 第 1 轮后 | 0.5725 | 0.5087 | 0.4584 | 0.3857 |
| epoch10.pt | 第 11 轮后 | 0.6809 | 0.5954 | 0.5973 | 0.5665 |
| epoch20.pt | 第 21 轮后 | 0.7194 | 0.6142 | 0.6598 | 0.6171 |
| epoch30.pt | 第 31 轮后 | 0.7939 | 0.7382 | 0.6798 | 0.6887 |
| epoch40.pt | 第 41 轮后 | 0.8072 | 0.7888 | 0.6672 | 0.6920 |
| epoch50.pt | 第 51 轮后 | 0.8052 | 0.7686 | 0.6720 | 0.6916 |
| epoch60.pt | 第 61 轮后 | 0.8092 | 0.7889 | 0.6833 | 0.7041 |
| epoch70.pt | 第 71 轮后 | 0.8019 | 0.7923 | 0.6812 | 0.7019 |
| epoch80.pt | 第 81 轮后 | 0.8145 | 0.7873 | 0.6844 | **0.7058** |
| epoch90.pt | 第 91 轮后 | **0.8158** | 0.7881 | 0.6803 | 0.7022 |

重要结论：

- `best.pt` 是按语义 mIoU 选出的第 61 轮状态；
- 每 10 轮快照中，最高 Macro-F1 出现在 `epoch80.pt`，为 0.7058；
- 最高 Accuracy 出现在 `epoch90.pt`，为 0.8158；
- 三者差距很小，但类别取舍不同；
- 说明“以包含 background 的 mIoU 保存 best”与论文关心的图像级 Macro-F1 并不完全一致；
- 后续必须把图像级 Macro-F1/少数类 Recall 纳入 checkpoint 选择，而不是只依赖 mIoU。

当前仍建议把 `best.pt` 作为 runs4 的官方语义基线，因为它是预先设定指标的最优点，并且 akiec、bcc 精度/召回组合较好；`epoch80.pt` 可作为诊断型候选保留，不应删除。

---

## 18. 存档、断电恢复与当前 checkpoint 状态

目录：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\weights`

已保留：

- `best.pt`；
- `last.pt`；
- `epoch0.pt`；
- `epoch10.pt`；
- `epoch20.pt`；
- `epoch30.pt`；
- `epoch40.pt`；
- `epoch50.pt`；
- `epoch60.pt`；
- `epoch70.pt`；
- `epoch80.pt`；
- `epoch90.pt`。

训练期间 `last.pt` 每个完整 epoch 更新，`save_period=10` 每 10 轮保存完整快照。本次结果中的训练计时在第 51 和第 58 轮附近发生两次重置，并且 `args.yaml` 记录了 resume，说明断点恢复机制在实际训练中已经生效。

但是早停完成后，Ultralytics 会剥离 `best.pt` 和 `last.pt` 中的优化器状态；它们现在约 12.59 MB，而周期快照约 49.95 MB。因此：

- 训练中断但尚未结束时：可运行 `tools\resume_yolo26_sem_v4.py` 从 last.pt 恢复；
- 当前 runs4 已正常早停完成：不应再把 `last.pt` 当作完整训练状态继续同一实验；
- `epoch90.pt` 仍可能含优化器，但其早停计数已接近终点，强行 resume 大概率很快再次停止；
- 新实验应写入 `experiments\05_...`、`experiments\06_...`，不覆盖第四次实验；
- 若以 runs4 权重微调，使用 `best.pt` 作为 pretrained weights 开一个新 run，而不是 `resume=True`。

---

## 19. 当前模型与项目的主要缺点

### 19.1 模型选择指标错位

当前 best.pt 按含 background 的 mIoU 选择，而论文最关心的是疾病诊断 Macro-F1、少数类 Recall 和 mel→nv。background 占 74.1% 像素，导致语义指标对真实目标不够敏感。

### 19.2 图像级诊断规则过于粗糙

runs4 目前用“最大疾病像素面积”得到图像类别。这不是经过专门训练的分类头，无法利用全局上下文、病灶内部特征分布和类别置信度，也容易让面积较大的错误区域主导结果。

### 19.3 仍存在明显类别混淆

主要混淆集中在 nv、mel、bkl；akiec 常被判为 bkl；df 常被判为 nv 或 bkl。模型并没有真正学会所有病理差异。

### 19.4 过拟合仍然明显

温和增强和像素权重减轻了部分问题，但 val CE 从中期开始持续上升，说明有限独立病灶和域偏差仍在被记忆。

### 19.5 单随机种子不足

目前核心实验主要使用 seed=42。df、vasc 样本很少，一次 seed 的提升可能是偶然。

### 19.6 语义与实例基线不是完全同构

runs3 与 runs4 同时改变了 YOLO 版本和任务头。虽然图像级指标可比较，但无法严格断言提升来自 C2PSA、YOLO26 主干还是语义任务形式。

### 19.7 尚未有真正的自定义注意力贡献

runs4 使用的是 YOLO26 原生 C2PSA，只能作为“现有注意力基线”。论文题目要求“基于注意力机制改进”，还需要自定义模块、插入位置说明、参数/FLOPs对比和消融实验。

### 19.8 尚未完成概率校准

没有 temperature scaling、ECE、Brier score、类别阈值或风险敏感决策。医疗识别不仅要预测对，还要知道置信度是否可信。

### 19.9 尚未完成外部泛化验证

单一 HAM10000 域可能让模型学习设备和来源特征。当前不能证明对其他医院或设备有效。

### 19.10 异常 mask 与 hard cases 尚未完成系统复核

少量异常掩码目前按用户决定保留。最终论文前应至少对高损失、跨模型一致误判和 mask 边界异常样本做一次人工清单审查。

---

## 20. 项目目前正在执行的计划

训练已经结束，当前没有 Python 训练进程。runs4 训练后审计、项目备份、目录整理、旧数据和无效环境清理已经完成。当前应进入 **统一评价器修正与第五次实验设计**：

1. 修复 `tools\train_skin7_yolo26_sem.py` 的 standalone val/test background=7 配置；
2. 建立统一评价脚本和与论文目标一致的 checkpoint 选择规则；
3. 设计第五次实验：只增加普通辅助图像分类头；
4. 后续再依次进行标准 CBAM 和提出注意力模块的控制实验；
5. 最终模型固定前不继续运行 test。

此前 runs4 训练后审计的逐类 IoU、图像级 P/R/F1、快照评价和 runs3 对比均已完成；不要重复这些耗时分析，除非评价代码发生变化。

---

## 21. 推荐的未来实验路线

### 阶段 A：先把评价体系修正好，不训练新模型

1. 项目中已经加入显式 8 类配置：`experiments\04_YOLO26s_Semantic_C2PSA\data_semantic_eval_v4.yaml`；
2. 修改 `tools\train_skin7_yolo26_sem.py` 的 standalone val/test，使其使用 background=7；
3. 建立一个统一评价脚本，同时输出：
   - 8 类 mIoU；
   - 前景七类 mean IoU；
   - 每类 IoU；
   - 图像级 Accuracy、Macro-P/R/F1；
   - confusion matrix；
   - mel→nv、nv→mel、akiec/bcc/df Recall；
   - 参数量、GFLOPs、推理速度；
4. 把评价从 image 级扩展到 lesion 级，多个同 lesion 图片不能被当作完全独立样本；
5. 对 val 做 lesion bootstrap 置信区间；
6. 定义主指标：建议以 Macro-F1 为主，以前景 mIoU 为约束，并单独报告医疗风险错误 `mel→nv`。

### 阶段 B：建立公平的多任务基线 runs5

目标：验证显式图像级分类头是否能解决“分割好但类别仍错”的问题。

建议结构：

- 保留 YOLO26s-sem 主干与语义头；
- 从中高层特征做全局池化；
- 增加 7 类辅助分类头；
- 语义 loss 保持 runs4 不变；
- 分类 loss 使用基于独立 lesion 数而非增强图片数的 class-balanced focal 或 balanced softmax；
- 图像级最终预测直接使用分类头概率，不再使用最大像素面积；
- 数据、seed、imgsz、batch、optimizer、增强全部与 runs4 保持一致。

这是必要控制实验，因为如果不做 runs5，后续加入注意力和分类头后无法判断提升来自哪个组件。

### 阶段 C：标准注意力对照 runs6

在 runs5 基础上加入标准 CBAM，建议只放在 1~2 个关键 neck 融合节点，避免到处堆叠：

- 通道注意力强调颜色/纹理通道；
- 空间注意力强调病灶区域；
- 记录参数量与 GFLOPs 增量；
- 其他设置完全不变。

CBAM 是已知方法，作用是提供一个可解释的注意力对照，不应直接作为论文最终创新点。

### 阶段 D：提出论文核心模块 runs7

建议的研究方向：**病灶掩码引导的多尺度双注意力模块**。

核心思想：

1. 从语义分割分支得到 foreground 概率；
2. 用 foreground 概率作为软空间注意力，对 P3/P4/P5 多尺度特征做 mask-guided pooling；
3. 使用轻量通道注意力（如 ECA 风格）选择与皮肤颜色、纹理、血管和边界相关的通道；
4. 融合多尺度病灶特征送入 7 类图像级分类头；
5. 增加分类与语义预测的一致性约束，但不要一开始同时修改太多 loss。

这个设计比单纯 CBAM 更贴合当前诊断：runs3 已证明定位不是主要瓶颈，真正需要的是“利用已定位病灶，引导分类特征聚合”。

### 阶段 E：损失与采样优化，单独做消融

注意力结构确定后再做，避免与结构改进混在同一实验里：

- lesion-balanced sampler，按原始 lesion 采样而不是按增强图片采样；
- class-balanced focal / balanced softmax；
- 对 mel→nv 设置经过预先声明的风险敏感代价；
- 对 df/akiec 使用适度召回权重；
- 不再继续增加离线增强总量；
- hard negative mining 重点放在 mel-vs-nv、mel-vs-bkl、akiec-vs-bkl，而不是重复所有误判样本。

### 阶段 F：最终可靠性实验

1. 筛选阶段统一 seed=42，快速排除无效结构；
2. 最终 2 个候选至少运行 3 个 seed；
3. 报告 mean±std；
4. 使用 lesion 级 bootstrap 95% CI；
5. 进行模型校准：temperature scaling、ECE、Brier score；
6. 给出 Grad-CAM/attention map 或病灶引导热力图；
7. 最终方法冻结后再使用 test；
8. 若可能，增加外部数据验证。

---

## 22. 建议的 runs5~runs8 实验矩阵

| Run | 主要变化 | 数据 | 目的 | 其余控制 |
|---|---|---|---|---|
| runs4 | YOLO26-sem 原生 C2PSA | V3 | 当前语义注意力基线 | 已完成 |
| runs5 | +普通辅助分类头 | V3 | 分离“显式分类头”的收益 | 与 runs4 一致 |
| runs6 | runs5 + CBAM | V3 | 通用注意力对照 | 与 runs5 一致 |
| runs7 | runs5 + 病灶引导多尺度双注意力 | V3 | 论文提出方法 | 与 runs5 一致 |
| runs8 | runs7 + lesion-balanced/loss 优化 | V3 | 优化策略消融 | 结构固定 |

不要在 runs5 同时改变数据、模型、注意力、loss 和采样，否则论文无法归因。

---

## 23. 下一轮训练建议参数

基于 runs4 曲线，不建议继续 150 epochs/patience30：

- `epochs: 100`；
- `patience: 20`；
- `batch: 16`；
- `imgsz: 640`；
- `optimizer: AdamW`；
- `lr0: 0.001` 作为首个控制实验；
- `seed: 42`；
- `deterministic: true`；
- 保持 runs4 的温和增强；
- `save_period: 10`；
- 新目录 `experiments\05_...`；
- 不使用 `exist_ok: true` 覆盖旧目录；
- 不使用 test 做早停或阈值调节。

注意：如果自定义模块使收敛速度变化，可以保留 max 120，但早停应监控与论文目标一致的复合指标，而不是只看含 background 的 mIoU。

---

## 24. 推荐的模型选择逻辑

建议不要把所有目标强行压成一个容易掩盖问题的单值。采用主指标+约束：

- 主指标：image-level Macro-F1；
- 约束 1：foreground mean IoU 不低于 runs4 的 0.5388 超过允许误差；
- 约束 2：mel→nv 不高于 runs4 的 18/167；
- 约束 3：akiec、bcc、df 的 Macro-Recall 高于基线；
- 次指标：参数量、GFLOPs、速度；
- 最终模型从 Pareto 最优候选中选，而不是只看 Accuracy。

若必须使用复合分数，可以预先定义，例如：

`Score = 0.50 × Macro-F1 + 0.25 × foreground-mIoU + 0.25 × minority-recall`

但权重必须在实验前固定，并单独报告每一项，不能在看完结果后反复改权重。

---

## 25. 论文中建议呈现的图表

1. V1/V2/V3 的图片数与独立 lesion 数对比；
2. 每类离线增强比例；
3. 数据源、dx_type 与类别的分布；
4. runs3 与 runs4 图像级 P/R/F1 对比；
5. runs4 训练/验证 CE、Dice、mIoU 曲线；
6. runs4 逐类 IoU；
7. runs3/runs4 混淆矩阵；
8. mel→nv、nv→mel 等关键错误变化；
9. runs4/runs5/runs6/runs7 消融表；
10. 参数量、GFLOPs、速度与性能折中图；
11. 注意力热力图/病灶引导图；
12. 多 seed 均值与置信区间。

---

## 26. 关键文件索引

### 数据

- 当前数据 YAML：`Data\processed_skin7_v3\data.yaml`
- 数据摘要：`Data\processed_skin7_v3\summary.json`
- 数据审计：`Data\processed_skin7_v3\metadata\v3_audit.json`
- V3 报告：`experiments\03_YOLO11s_V3_NV3200\DATA_V3_REPORT.md`
- V3 构建脚本：`tools\prepare_skin7_dataset_v3.py`

### 第三次实验（原 runs3）

- 目录：`experiments\03_YOLO11s_V3_NV3200`
- 权重：`experiments\03_YOLO11s_V3_NV3200\results\train\weights\best.pt`
- 额外诊断：`experiments\03_YOLO11s_V3_NV3200\analysis`

### 第四次实验（原 runs4）

- 目录：`experiments\04_YOLO26s_Semantic_C2PSA`
- 最佳语义权重：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\weights\best.pt`
- 第 81 轮诊断候选：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\weights\epoch80.pt`
- 训练曲线：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\results.csv`
- 图像级结果：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\val_image_level`
- 快照图像级结果：`experiments\04_YOLO26s_Semantic_C2PSA\results\semantic\skin7_yolo26s_sem_balanced\val_image_level_epochs`
- 修正后语义验证图：`experiments\04_YOLO26s_Semantic_C2PSA\results\analysis\corrected_semantic_val`

### 脚本

- runs4 训练：`tools\train_skin7_yolo26_sem.py`
- runs4 配置：`experiments\04_YOLO26s_Semantic_C2PSA\train_yolo26s_sem_v4.yaml`
- 语义独立评价配置：`experiments\04_YOLO26s_Semantic_C2PSA\data_semantic_eval_v4.yaml`
- 训练中断恢复：`tools\resume_yolo26_sem_v4.py`
- 图像级评价：`tools\evaluate_yolo26_sem_image_level.py`
- 项目索引：`PROJECT_INDEX.md`
- 清理报告：`docs\CLEANUP_REPORT_2026-07-20.md`

---

## 27. 新对话必须遵守的事项

1. 不删除或覆盖 `experiments\01_...` 至 `experiments\04_...`。
2. 后续实验从 `experiments\05_...` 开始编号。
3. 不修改 V3 的 val/test 内容。
4. 最终方法冻结前不继续使用 test 指标。
5. 语义 standalone validation 必须显式设置 background=7。
6. 不把 semantic mIoU 与 instance Mask mAP 直接比较。
7. 不能只看 Accuracy 或 overall pixel accuracy。
8. attention 改进必须有无注意力/标准注意力/提出模块的控制实验。
9. 每次实验只改变计划中的一个主要变量。
10. 用户希望训练从 PyCharm 中启动，以便看到实时参数和曲线；除非用户明确要求，不要在隐藏终端启动长时间正式训练。
11. 从 `E:\App\DaiMa\JetBrains\Project\BeiFen\PiFuV2` 恢复文件时只能复制，不能剪切或移动备份原件。

---

## 28. 可直接粘贴到新对话的开场提示

```text
我的项目在 E:\App\DaiMa\JetBrains\Project\PiFuV2，毕业论文题目是《基于注意力机制改进YOLO的皮肤病识别算法研究》。请先读取 PROJECT_INDEX.md，再完整读取 docs\PROJECT_HANDOFF_2026-07-20.md 和 docs\CLEANUP_REPORT_2026-07-20.md，然后检查其中列出的第四次实验结果和关键文件，不要重新猜测历史上下文。

当前固定数据是 Data\processed_skin7_v3\data.yaml，类别顺序为 akiec、bcc、bkl、df、mel、nv、vasc。runs4 使用 YOLO26s-sem，在第91轮早停，语义 best 是第61轮，正确 mIoU=0.59122；图像级 Accuracy=0.8092、Macro-F1=0.7041。它提高了 bcc/vasc，mel→nv 从26降到18，但 df Recall 降到0.333，nv→mel 增到84。standalone semantic val 存在 background 被映射到 vasc 的问题，必须使用显式 nc=8、bg_class_idx=7 的评价 YAML。

接下来先不要运行 test，也不要删除或覆盖 experiments\01~04。请先完成评价器修正和第五次实验的详细技术设计：第五次只增加普通辅助图像分类头，作为后续 CBAM 和病灶掩码引导多尺度双注意力的公平控制实验。所有正式训练仍从 PyCharm 启动，结果继续按 experiments\05、06、07 保存。从备份恢复文件时只能复制，不能剪切。
```

---

## 29. 最终结论

runs4 的训练本身是成功且可用的：早停正常、权重完整、断点恢复有效、语义分割性能稳定、模型更轻量，也在 bcc、vasc 和 mel→nv 上取得了有意义的改进。但它仍存在分类过拟合、df/akiec 召回不足、nv→mel 增加、像素指标被 background/NV 支配、图像级决策粗糙和 checkpoint 选择指标错位等问题。

因此当前最正确的决策不是继续堆训练轮数，也不是再盲目扩大数据增强，而是将 runs4 固定为 YOLO26 原生注意力基线，建立显式图像级诊断头，再用严格消融验证标准注意力与提出的病灶引导注意力模块。只有这样，后续提升才能成为一篇可信的“基于注意力机制改进 YOLO”的毕业论文结果。
