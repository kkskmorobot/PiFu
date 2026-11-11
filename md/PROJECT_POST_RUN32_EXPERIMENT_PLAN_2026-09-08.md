# Run32 之后的实验路线规划（2026-09-08）

`PLAN_STATUS=TEST1_FINAL_EVALUATION_COMPLETE_DEVELOPMENT_CLOSED`

## 1. 目的与论文主线
训练文件路径：E:\App\DaiMa\JetBrains\Project\PiFuV2\tools

md文件路径：E:\App\DaiMa\JetBrains\Project\PiFuV2\md

结果文件路径：E:\App\DaiMa\JetBrains\Project\PiFuV2\experiments

本文件记录 Run32 完成后的实验顺序和停止条件，防止后续上下文压缩造成规划丢失。

论文主线统一为：

> YOLO26s 用于低成本的前期方法探索；YOLO26m 作为正式主模型，在统一的二值语义分割、八类图像分类和 partial-label 异构监督体系下完成注意力消融、风险优化与最终模型选择。

YOLO11s 只作为必要的对比模型，不作为论文改进主线。Run1～Run32 的历史、失败结果和封存状态全部保留，不删除、不覆盖、不改写。

## 2. 已经在 YOLO26m 上完成、无需重复补做的内容

以下设计已经成为 YOLO26m 正式实验的公共底座，不能再误认为只在 YOLO26s 上做过：

- 二值病灶语义分割与八类图像分类双任务；
- IMA++ 分割标签与 ISIC2019 分类标签组成的 partial-label 异构监督；
- 缺失标签屏蔽、统一 development train/val、冻结 taxonomy/split/manifest/mask；
- P3/P4 多尺度特征的全局汇聚分类头；
- 统一的 Joint、Group Macro-F1、Group Recall、MEL→NV、malignant→benign、校准指标、延迟早停和 checkpoint bundle 合同。

已经形成的 YOLO26m 注意力对照为：

| 组别 | 已完成实验 | 作用 |
|---|---|---|
| Plain P3/P4 | Run18、Run22、Run24 | 三个 seed 的无注意力基线 |
| P3/P4 ECA | Run21、Run23、Run25 | 三个 seed 的通道注意力对照 |
| P3/P4 ECA + spatial7 | Run27、Run28、Run26 | 三个 seed 的通道与空间注意力组合 |
| BLG-MDA | Run16、Run19、Run20 | 病灶引导多尺度注意力的迁移与演进证据 |

因此，后续不再机械地把所有 YOLO26s 旧实验重新训练一遍。真正尚缺的注意力对照是：在当前 YOLO26m 正式底座上进行一次严格的标准 CBAM 单因素实验。

## 3. Run32 完成后首先关闭风险辅助头分支（已完成）

Run32 训练完成后，先检查训练完整性、CSV 连续性、最佳 Joint bundle、checkpoint 大小与 SHA-256、`_SUCCESS`、fingerprint、恢复状态，以及是否误用 test/external。随后只与同 seed43 的 Run22 plain 基线进行预注册配对比较。

### 3.1 Run32 达到预注册门

如果 MEL→NV 和 malignant→benign 均有实质改善，并且 Group Macro-F1 与 Joint 没有明显损害：

1. 以完全相同的恶性/良性辅助头和固定权重，分别在 seed42、seed44 上做两次配对复现；
2. 每个 seed 分别与 Run18、Run22、Run24 的对应 plain 基线比较；
3. 汇总三 seed 均值、离散程度、改善方向一致性和正式 feasible checkpoint 数量；
4. 三 seed 证据成立时，辅助头才能进入最终候选；否则仍判定为不稳定，不继续调权重或改结构追逐 validation。

### 3.2 Run32 未达到预注册门（已完成）

如果风险改善不足，或明显损害 Group Macro-F1/Joint：

1. 停止该辅助头，不做 seed42/44 复现；（已完成）
2. 连同 Run29、Run30、Run31 的结果，结束训练期风险损失/辅助头探索路线；（已完成）
3. 不根据 Run32 validation 结果临时修改辅助权重、类别定义或隐藏层后继续试验；（已完成）
4. 后续风险控制转入第 6 节的固定代价决策研究。（已完成）

## 4. 补齐 YOLO26m 标准 CBAM 对照（已完成）

Run32 分支关闭后，在当前 YOLO26m plain P3/P4、partial-label 公共底座上补做标准 CBAM 单因素对照。首轮使用 seed43，与 Run22 配对；除 CBAM 外，数据、损失、采样、训练生命周期和模型选择规则全部保持一致。

该实验的论文目的不是继续堆叠模块，而是补齐以下清晰的注意力消融链条：

> Plain → 标准 CBAM → ECA → ECA + spatial7 → BLG-MDA

### 4.1 标准 CBAM 首轮有明确价值

如果 seed43 相对 Run22 在分类、Group 指标或风险上出现预注册的明确改善，同时没有明显破坏分割与 Joint：

1. 再补 seed42、seed44 的同构配对复现；
2. 与已有的三 seed Plain、ECA、ECA+spatial7 进行统一统计比较；
3. 只有三 seed 结果支持时，标准 CBAM 才能成为最终注意力候选。

### 4.2 标准 CBAM 首轮无明确价值（已完成）

如果 seed43 结果为负或仅有很小、互相冲突的变化，则停止 CBAM 复现，不再增加 seed42/44。该单 seed 结果作为“标准 CBAM 在 YOLO26m 公共底座上的对照”写入论文即可。（已完成）

## 5. 锁定 YOLO26m 注意力主线（已完成）

完成第 3、4 节后，统一比较 Plain、ECA、ECA+spatial7、BLG-MDA、标准 CBAM，以及通过复现门的 Run32 辅助头候选。选择时不得只看某个 seed 的最高 Joint，而应同时考虑：

- 三 seed 平均表现与方向一致性；
- Binary IoU、Image Macro-F1、Group Macro-F1、Group Recall 和 Joint；
- MEL→NV 与 malignant→benign 风险；
- feasible checkpoint 数量；
- 参数增量、结构复杂度和论文可解释性。

当前 ECA+spatial7 只能称为“风险和 recall 方向有希望、总体 Joint 接近持平”的候选，不能预先宣布为最终模型。若所有注意力方法的多 seed 证据都不足，可诚实地锁定 plain YOLO26m 为最稳健基线，并把注意力结果报告为有限或不稳定增益。

架构一旦锁定，停止继续组合新的注意力模块，避免把论文变成无边界的试错记录。（已完成）

锁定结论：Plain YOLO26m P3/P4 partial-label 多任务架构作为最终训练主线，三 seed 证据为 Run18/Run22/Run24。（已完成）

## 6. 固定架构后的临床风险决策实验（已完成）

只有在第 5 节锁定模型架构和 checkpoint 选择规则后，才进入 development-only 的固定代价决策研究。该阶段研究分类决策规则，不再同时改变训练结构。

执行原则：

1. 事先固定恶性/良性代价和决策规则，再查看正式比较结果；（已完成）
2. 使用冻结模型输出进行单因素比较，不反复根据同一 validation 结果追调代价；（已完成）
3. 比较原始八类 argmax 与固定代价决策在 Group Macro-F1、Group Recall、MEL→NV、malignant→benign 和 Joint 上的变化；（已完成）
4. 只有风险明显下降且总体效用损害在预注册范围内，决策规则才进入最终候选；（已完成，接受门未通过，规则未进入最终候选）
5. 如果仍然没有正式 feasible checkpoint，则停止继续追逐风险门，明确报告约束未满足，并把它列为研究局限，不能把不合格结果包装成最终可行模型。（已完成）

### 6.1 Val1 固定代价决策结论（已完成）

本次任务的规范名称为 **Val1**，不是新的模型训练 Run34。其预注册、程序和物理结果在执行时使用了历史名称 Run34。完成后用户已把 `tools\run34` 的全部内容移动到 `tools\val1` 并删除原工具目录；冻结文件仍保留原来的 Run34 文件名和内部历史路径，避免静默改写既有 SHA-256 与 fingerprint。`experiments\34_...` 完成结果和失败 partial 保持不动，后续论文与汇总统一称为 Val1，并显式注明历史预注册名。

Val1 复用了 Run18、Run22、Run24 的冻结 best-Joint bundle，全程未训练，也未访问 test、external 或 PAD-UFES-20。固定代价规则在三个 seed 上均降低了 MEL→NV 和 malignant→benign，三 seed 平均绝对变化分别为 `-0.20748` 和 `-0.14654`；但平均 Group Macro-F1 为 `-0.04891`、平均 Joint 为 `-0.02798`，且 seed42/seed43 的 Group Macro-F1 分别下降 `-0.06699`/`-0.07780`，未通过预注册效用保护门。

最终决策：拒绝固定代价规则，保留原始八类 argmax；停止在同一 validation 上继续调整代价、阈值或决策规则。正式 feasible checkpoint 仍为 0，作为研究局限报告。

## 7. 最终模型锁定、一次性评估与论文汇总

开发阶段结束后，先完成以下事项：

1. 冻结最终架构、权重 bundle、推理决策规则、输入尺寸、taxonomy 和全部 fingerprint；（已完成）
2. 汇总三 seed 主实验、均值/离散程度、成对差值、失败实验和模型复杂度；（已完成；参数量已按冻结预检报告汇总，现有日志未统一保存自定义头总 GFLOPs，因此只报告共同语义主干约 59.1 GFLOPs，不编造跨结构差值）
3. 生成论文所需的分割示例、分类混淆矩阵、恶性/良性风险分析、校准分析和消融表；（已完成 development 版本）
4. 明确区分 YOLO26s 的前期筛选、YOLO26m 的正式验证，以及探索性风险实验；（已完成）
5. 最终候选锁定后，不再依据任何最终评估结果返回开发阶段修改模型。（已完成；Test1 后 development 永久关闭）

最终研究候选锁定说明见 `tools\val1\PROJECT_FINAL_MODEL_LOCK_2026-09-11.md`。该锁定不代表模型满足临床风险约束，也不构成读取 test/external 的授权。

Development 汇总、机器可读表、工作簿与论文图件见 `tools\sum1`。汇总仅使用 Run16～Run33 的 development `results.csv` 与 Val1 冻结预测，没有访问 test/external/PAD-UFES-20。

IMA++ test、ISIC2018 external 已在独立预注册、发布门通过并获得用户新的明确确认后完成一次性最终评估。正式结果见 `experiments\test1\results\result.json`，摘要见 `tools\test1\result_summary.md`。一次性授权已经消费，禁止覆盖、重跑或依据结果返回 development 修改模型。PAD-UFES-20 未使用并继续排除。（已完成）

## 8. 预计剩余训练规模与停止条件

Run32 之后的新增正式训练数量取决于条件门：

- Run32 失败：不增加辅助头复现；Run32 成功：增加两个 seed 复现；
- 标准 CBAM 必做一个 seed43 严格对照；只有首轮有明确价值时才增加两个 seed 复现；
- 固定代价决策研究原则上复用冻结 checkpoint，不应演变成大量重新训练；
- 完成上述两条分支后必须进入模型锁定和论文汇总，不再无限增加注意力/损失组合。

因此，Run32 之后预计还需要 **至少 1 次、最多 5 次** 新的模型训练；实际数量由 Run32 和标准 CBAM 的预注册结果门决定。任何后续正式 run 仍必须逐次完成预注册、静态/GPU/恢复预检、发布门和用户明确确认，并由用户在 PyCharm 锁定入口点击启动，Codex 不得代替用户启动训练。

实际结果：Run32 与 Run33 均未触发跨 seed 复现分支，Run32 之后仅新增 Run33 一次模型训练，已达到本路线的训练停止条件。（已完成）

Val1 随后完成了冻结 checkpoint 的 development-only 决策评估，不计作新增模型训练。Val1 未通过接受门，因此不存在下一次合规训练分支。Development 汇总与图表已在 `sum1` 完成；独立的 `test1` 一次性最终评估也已完成并关闭发布门。后续工作只允许进行结果汇总、论文写作与展示，不再训练、调参或重跑最终评估。（已完成）

## 9. 不可突破的边界

- 不得修改冻结的 manifest、split、mask、taxonomy 或既有实验结果；
- 不得删除、覆盖或重新编号 Run1～Run32；
- 不得把 seed 复现包装为新的结构创新；
- 不得在看到 validation 结果后无预注册地调权重、阈值或组合模块；
- 不得在最终模型锁定前运行 test/external；
- 每次实验坚持单一因素、独立 experiments 编号目录和完整 checkpoint/fingerprint 合同。
