# PiFuV2 Run10 Stage 2 前置状态

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`CHECKPOINT_BUNDLE_CONTRACT_V2_GATE=PASS`  
`CHECKPOINT_CPU_CONTINUITY_GATE=PASS`  
`CHECKPOINT_RTX5080_CONTINUITY_GATE=PASS`  
`CHECKPOINT_RTX5080_AMP_WORKERS_GATE=PASS`  
`EXTERNAL_EVALUATION_BOUNDARY_PROTOCOL_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_SOURCE_AUDIT_GATE=PASS`  
`PREFERRED_EXTERNAL_CLASSIFICATION_CANDIDATE=PAD-UFES-20_V1`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=BLOCKED_PENDING_DOWNLOAD_HASH_AND_STRICT_DEDUP`  
`RUN10_PREREGISTRATION_GATE=BLOCKED_PENDING_USER_APPROVAL_AND_FACTOR_LOCK`  
`TEST_METRICS_USED=FALSE`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

本文件是当前最新状态记录，但不是 Run10 放行文件。没有创建
`experiments\10_...`，没有正式训练，没有 test/external 指标。

## 已完成

- 删除损坏隔离 ZIP
  `Data\1\ISIC_2019_Training_Input.corrupt_dualwriter_20260804_202621.zip`；正确官方
  ZIP 保留，大小 9,771,618,190 bytes。
- checkpoint bundle 升级到 v2，保存并恢复未量化 live model、EMA、optimizer、
  GradScaler、scheduler、EarlyStopping、专项选择状态、完整 RNG 和数据进度。
- SIGINT 在完整 epoch 边界请求停止；`resume=auto` 只选择含 `_SUCCESS`、文件哈希
  正确且 fingerprint 匹配的最新 bundle。
- 单卡每 epoch 使用 seed/rank/epoch 派生 RNG 并重建 iterator，消除 Windows
  `InfiniteDataLoader` 连续/恢复排列差异。
- 9 个 unittest 全部通过。
- CPU、RTX 5080 AMP-off、RTX 5080 AMP-on + workers=2 三套端到端连续/恢复
  preflight 均 PASS，最终实际最大绝对差均为 0。
- 外部域、IMA++ blind test 与分类外部候选的角色边界已写入
  `RUN10_EXTERNAL_AND_BLIND_EVALUATION_PROTOCOL_2026-08-04.md`。
- 官方来源候选审计首选 PAD-UFES-20；尚未下载，映射、局限性和严格去重要求见
  `RUN10_EXTERNAL_CLASSIFICATION_CANDIDATE_AUDIT_2026-08-04.md`。

## 仍阻断

1. PAD-UFES-20 尚未取得用户下载确认，也未执行归档哈希和严格去重验收；
2. Run10 的唯一预注册因素、初始化、loss 权重、采样、优化器、预算、seed、
   early stopping 和模型选择规则尚未由用户确认；
3. 最新最终 resolution 尚未写出 `PROJECT_RUN10_GATE=PASS`；
4. 用户尚未明确授权设计并启动 `experiments\10_...`。

因此继续禁止正式训练和任何 test/external evaluation。
