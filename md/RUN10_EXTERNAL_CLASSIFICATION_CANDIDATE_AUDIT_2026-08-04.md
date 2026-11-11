# Run10 external classification 候选来源审计

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`PREFERRED_EXTERNAL_CLASSIFICATION_CANDIDATE=PAD-UFES-20_V1`  
`EXTERNAL_CLASSIFICATION_SOURCE_AUDIT_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_DOWNLOAD_GATE=BLOCKED_PENDING_USER_CONFIRMATION`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=BLOCKED_PENDING_DOWNLOAD_HASH_AND_STRICT_DEDUP`  
`EXTERNAL_CLASSIFICATION_EVALUATION_GATE=BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION`

本文件只确定首选候选与验收方法，不授权下载、训练、模型选择或 external/test
评价。

## 1. 首选：PAD-UFES-20

官方数据论文：`https://doi.org/10.1016/j.dib.2020.106221`  
数据 DOI：`https://doi.org/10.17632/zr7vgbcyr2.1`

PAD-UFES-20 包含 2,298 张由智能手机采集的临床图像，来自 1,373 名患者、
1,641 个病灶。它不是当前训练数据的皮肤镜采集域，适合用于 image-only 分类模型
的外部域压力测试。

预注册映射：

| PAD 官方诊断 | 数量 | taxonomy v1 | 说明 |
|---|---:|---|---|
| ACK | 730 | akiec | actinic keratosis |
| BCC | 845 | bcc | basal cell carcinoma |
| MEL | 52 | mel | melanoma |
| NEV | 244 | nv | melanocytic nevus |
| SCC | 192 | scc | squamous cell carcinoma |
| SEK | 235 | bkl | seborrheic keratosis -> benign keratosis |

PAD-UFES-20 不覆盖 df、vasc，因此将同时报告：

- full-8 macro precision/recall/F1，df 与 vasc support=0；
- supported-6 macro precision/recall/F1；
- 六类逐类指标及 support；
- mel->nv、nv->mel、malignant(mel/bcc/scc)->benign；
- NLL、Brier、ECE，但不在该外部集上拟合温度或阈值。

局限性必须写入最终报告：全数据仅约 58.4% 病灶经活检确认；三类皮肤癌为
100% 活检确认，而 ACK、NEV、SEK 的确认率较低。外部结果只能解释为临床智能手机
域的候选验证，不能替代 IMA++ internal blind test。

## 2. 下载后必须执行的验收

1. 固定数据版本、下载 URL、许可文本、归档大小和 SHA-256；
2. 校验 metadata 的 2,298 图像、1,641 lesion、1,373 patient 及六类计数；
3. 只使用官方 `diagnostic` 字段；未知/空标签排除，禁止猜测；
4. 按 patient ID 和 lesion ID 建立 group manifest；
5. 对以下完整参考池执行 ID、文件 SHA-256、尺寸、严格 `pHash+dHash` 去重：
   - ISIC 2019 官方 training；
   - HAM10000；
   - IMA++ 原始 14,967 ID 和 binary v1；
   - ISIC 2018 Task 1 external segmentation candidate；
6. 同一文件/严格感知组若出现冲突诊断，整组隔离，不多数票；
7. 输出 manifest、excluded、duplicate groups、hashes、provenance、报告和
   `_SUCCESS`；
8. 数据只放 external candidate 命名空间，不进入 combined train/val YAML。

## 3. 未采用候选

- BCN20000：它是 ISIC 2019 training 的组成来源，已被当前分类训练池覆盖，不能
  作为独立外部数据。
- Derm7pt：具有临床/皮肤镜多模态与 1,011 lesion cases，但诊断层级映射、潜在
  来源重叠和下载许可仍需逐项审计；当前不优先于 PAD-UFES-20。

