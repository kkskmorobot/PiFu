# PiFuV2 Run10 Stage 3 前置状态

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`CLASSIFICATION_TAXONOMY_GATE=PASS`  
`CLASSIFICATION_DATA_FILE_DEDUP_GATE=PASS`  
`CLASSIFICATION_INCREMENTAL_DATASET_GATE=PASS`  
`PARTIAL_LABEL_LOADER_GATE=PASS`  
`MULTITASK_MODEL_IMPLEMENTATION_GATE=PASS_WITH_CPU_AND_SINGLE_BATCH_GPU_SMOKE`  
`CHECKPOINT_BUNDLE_CONTRACT_V2_GATE=PASS`  
`CHECKPOINT_CPU_CONTINUITY_GATE=PASS`  
`CHECKPOINT_RTX5080_CONTINUITY_GATE=PASS`  
`CHECKPOINT_RTX5080_AMP_WORKERS_GATE=PASS`  
`UNIFIED_VAL_PROTOCOL_IMPLEMENTATION_GATE=PASS_STATIC_ONLY`  
`EXTERNAL_EVALUATION_BOUNDARY_PROTOCOL_GATE=PASS`  
`PAD_UFES20_SOURCE_DOWNLOAD_GATE=PASS`  
`PAD_UFES20_STRICT_DEDUP_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_EVALUATION_GATE=BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION`  
`RUN10_PREREGISTRATION_GATE=BLOCKED_PENDING_USER_APPROVAL_AND_FACTOR_LOCK`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

本文件是 2026-08-04 当前最新状态，但不是 Run10 放行文件。没有创建
`experiments\10_...`，没有正式训练，没有运行 external/test 评价。

## 本阶段完成

- PAD-UFES-20 v1 官方源下载、版本/许可/大小/SHA-256/ZIP CRC 与 metadata-ID
  合同全部 PASS；官方源位于
  `Data\external_candidates\pad_ufes_20_v1_official`。
- 对 ISIC 2019 official training、HAM10000、IMA++ raw、IMA++ binary v1 与
  ISIC 2018 external candidate 执行完整 ID/SHA/严格感知去重。
- 发布 `Data\processed_pad_ufes_20_external_classification_v1`：保留 2,269，
  排除 29，patient-lesion groups=1,876；独立逐图验证 PASS。
- PAD 只承担 external candidate 角色，没有进入 combined train/val YAML。
- 详细证据见 `PAD_UFES20_EXTERNAL_CLASSIFICATION_ACCEPTANCE_2026-08-04.md`。
- 清理错误 Range 合并物/分段 2,441,130,186 bytes，以及经修正版替代且核心哈希
  完全一致的 superseded 输出 3,532,122,554 bytes；共清理 5,973,252,740 bytes，
  删除不可恢复，正确官方源和正式修正版保留。
- 复核 `experiments`：`10_*` 目录数为 0；Data 下无 `.building` 或 superseded 残留。

## 保留边界

- Run1~Run9 全部保留；Run9 epoch 69 正常 early stopping 的既有事实未改动。
- 没有改动 `Data\processed_imaplusplus_binary_v1` 的 manifest、split、0/1 掩码
  值域或 test 规则。
- 没有改动 `Data\processed_isic2018_task1_incremental_v1` 的 89 对外部候选。
- 没有加载 IMA++ test，没有运行 PAD/ISIC2018 external 指标。
- 旧 V3 仍只用于历史复现。

## 仍阻断 Run10

1. Run10 的单一预注册因素、初始化权重、loss 权重、采样、优化器、训练预算、seed、
   early stopping 和模型选择规则尚未由用户确认并冻结；
2. 当前最新文档仍未写出 `PROJECT_RUN10_GATE=PASS`；
3. 用户尚未在 gate PASS 后明确授权设计并启动 `experiments\10_...`。

因此继续禁止设计/创建 Run10 实验目录、正式训练和任何 external/test evaluation。
下一步必须停在用户确认门：先由用户决定是否进入最终 factor-lock 与 Run10 放行审查。
