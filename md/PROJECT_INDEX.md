# PiFuV2 项目索引

项目路径：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

## 当前应使用的内容

- 有效虚拟环境：`.venv`
- 当前唯一推荐数据集：`Data\processed_skin7_v3`
- 当前数据配置：`Data\processed_skin7_v3\data.yaml`
- 最新已完成正式实验：`experiments\08_YOLO26s_Semantic_LesionBalancedSampler`（第 82 轮正常早停）
- 当前正式实验：`experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing`（2026-08-01 已从 PyCharm 启动）
- Run9 唯一变量：固定 Run8，只对图像分类 CE 使用 `label_smoothing=0.05`
- 详细交接文档：`docs\PROJECT_HANDOFF_2026-07-20.md`
- 公共脚本：`tools`
- 预训练模型：`models`

## 正式实验记录

1. `experiments\01_YOLO11s_V1_Baseline`
   - YOLO11s 实例分割初始基线
   - V1 数据，NV 上限约 3500
2. `experiments\02_YOLO11s_V2_DataBalance`
   - 扩大少数类平衡与 hard-example 增强
   - 第 58 轮早停，结果下降
3. `experiments\03_YOLO11s_V3_NV3200`
   - 回到稳定增强策略，只把 NV 调整到 3200
   - 当前 YOLO11s 稳定基线
4. `experiments\04_YOLO26s_Semantic_C2PSA`
   - YOLO26s 语义分割与原生 C2PSA 注意力基线
   - 第 91 轮早停，最佳语义状态为第 61 轮
5. `experiments\05_YOLO26s_Semantic_AuxClassifier`
   - 正式训练已完成：第 84 轮停止；最终 val 审计与 runs4 对比已完成
   - 相比实验 04 只增加 P3/P4 全局池化七分类辅助头
   - 最佳权重按预先固定的 Macro-F1、前景 mIoU、少数类 Recall 与 mel→nv 风险评分选择
6. `experiments\06_YOLO26s_Semantic_AuxClassifier_CBAM`（已完成；第 57 轮正常早停，最佳为第 37 轮）
   - 在 runs5 的 P3/8 neck 融合输出后只增加一个标准 CBAM
   - 其余数据、loss、采样、训练参数与 runs5 保持一致
7. `experiments\07_YOLO26s_Semantic_LesionGuidedDualAttention`（已完成；第 55 轮正常早停）
   - 从 runs5 基线出发，只替换图像分类聚合为病灶掩码引导的 P3/P4/P5 双注意力
   - 语义分支、数据、loss、采样和训练参数与 runs5 保持一致
   - 综合分最佳为第 35 轮，Macro-F1 最佳为第 43 轮；没有 checkpoint 满足全部预注册约束
   - 结论见 `RUNS7_FINAL_VAL_ANALYSIS_2026-07-30.md`
8. `experiments\08_YOLO26s_Semantic_LesionBalancedSampler`（已完成；第 82 轮正常早停）
   - 固定 Run7 模型、loss、优化器、增强、epoch 长度、验证和选择规则
   - 每类每轮配额保持为 Run7 的 800/1000/1400/320/1400/3200/392
   - 只在每类内部先均匀选 lesion，再均匀选该 lesion 的图像版本
   - 综合分最佳为第 62 轮，Macro-F1 最佳为第 79 轮；第 60 轮是唯一满足全部预注册约束的 checkpoint
   - 正式候选为 `best_feasible_macro_f1.pt`（第 60 轮）；没有运行 test
   - 方案与验收见 `RUN8_PROTOCOL_2026-08-01.md` 和 `VERIFICATION_2026-08-01.md`
9. `experiments\09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing`（训练中）
   - 固定 Run8 模型、sampler、类别配额、语义 loss、优化器、增强和选择规则
   - 唯一变化是图像分类 CE 使用 `label_smoothing=0.05`
   - RTX 5080 的 1% 训练 + 完整 1504 张 val 冒烟已通过
   - 2026-08-01 已通过 `Run9 Label Smoothing Formal` PyCharm 配置可见启动
   - 预注册方案见 `RUN9_PROTOCOL_2026-08-01.md`

未来实验从 `05_...` 开始编号，不要覆盖前四次结果。

## 工具脚本

- `tools\train_skin7.py`：YOLO11 实例分割训练入口
- `tools\train_skin7_yolo26_sem.py`：YOLO26 语义分割训练入口
- `tools\resume_yolo26_sem_v4.py`：仅用于 runs4 训练尚未结束时的中断恢复
- `tools\evaluate_yolo26_sem_image_level.py`：语义模型图像级七分类评价
- `tools\evaluate_skin7_unified.py`：统一输出语义、图像、lesion、bootstrap 与效率指标
- `tools\train_skin7_yolo26_multitask.py`：实验 05 的 PyCharm 训练入口
- `tools\resume_yolo26_multitask_v5.py`：实验 05 的 last.pt 断点续训入口
- `tools\skin7_multitask.py`：实验 05 的项目内模型、损失、验证器与预测器
- `tools\train_skin7_yolo26_cbam.py`：实验 06 的 PyCharm 训练入口
- `tools\resume_yolo26_cbam_v6.py`：实验 06 的 last.pt 断点续训入口
- `tools\skin7_cbam.py`：实验 06 的标准 CBAM、模型与 checkpoint 选择实现
- `tools\train_skin7_yolo26_lesion_guided.py`：实验 07 的 PyCharm 训练入口
- `tools\resume_yolo26_lesion_guided_v7.py`：实验 07 的完整 last.pt 断点续训入口
- `tools\skin7_lesion_guided.py`：实验 07 的病灶引导多尺度双注意力、模型与训练器
- `tools\train_skin7_yolo26_lesion_balanced.py`：实验 08 的 PyCharm 正式训练入口
- `tools\resume_yolo26_lesion_balanced_v8.py`：实验 08 的完整 last.pt 断点续训入口
- `tools\skin7_lesion_balanced.py`：实验 08 的固定类别配额、类内 lesion 均衡采样器
- `tools\train_skin7_yolo26_label_smoothed.py`：实验 09 的 PyCharm 正式训练入口
- `tools\resume_yolo26_label_smoothed_v9.py`：实验 09 的完整 last.pt 断点续训入口
- `tools\skin7_label_smoothed.py`：实验 09 的分类 CE label smoothing loss、模型与训练器
- `tools\prepare_skin7_dataset*.py`：V1/V2/V3 数据构建历史脚本

脚本已适配新的 `tools` 目录，项目根路径仍正确指向 PiFuV2。

## 数据说明

未来训练只需要 `Data\processed_skin7_v3`。旧 V1、V2 和原始下载数据若被删除，只能从用户保留的官方原始备份重新取得。V3 已包含可直接训练的 images、labels、metadata、data.yaml 和 train/val/test。

## 备份

非数据、非虚拟环境备份位于：

`E:\App\DaiMa\JetBrains\Project\BeiFen\PiFuV2`

备份共 268 个文件、2,310,555,920 字节，已完成逐文件 SHA-256 校验。校验清单为 `BACKUP_MANIFEST_SHA256.csv`。

## 重要纪律

- 不删除或覆盖四次正式实验。
- 最终模型确定前不要继续使用 test 指标。
- YOLO26 standalone semantic validation 必须显式使用 background=7 的 8 类配置。
- 不把语义 mIoU 与实例 Mask mAP 直接比较。
- 正式训练仍建议从 PyCharm 启动。
