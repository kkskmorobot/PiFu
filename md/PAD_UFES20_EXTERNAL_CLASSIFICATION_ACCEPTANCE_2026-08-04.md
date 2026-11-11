# PAD-UFES-20 外部分类候选 v1 验收

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`PAD_UFES20_SOURCE_DOWNLOAD_GATE=PASS`  
`PAD_UFES20_SOURCE_HASH_AND_ARCHIVE_GATE=PASS`  
`PAD_UFES20_METADATA_CONTRACT_GATE=PASS`  
`PAD_UFES20_STRICT_DEDUP_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_EVALUATION_GATE=BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION`  
`TEST_METRICS_USED=FALSE`  
`EXTERNAL_METRICS_USED=FALSE`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

本验收只放行 PAD-UFES-20 的 external candidate 数据角色，不授权训练、模型选择、
external evaluation 或 IMA++ internal blind test。

## 1. 官方来源与许可

- 官方数据页：`https://data.mendeley.com/datasets/zr7vgbcyr2/1`
- 数据 DOI：`10.17632/zr7vgbcyr2.1`
- 数据论文 DOI：`10.1016/j.dib.2020.106221`
- 固定版本：v1；许可：CC BY 4.0。
- 官方源目录：`Data\external_candidates\pad_ufes_20_v1_official`
- 来源清单：`SOURCE_MANIFEST.json`，SHA-256
  `b6069917e442db8cadd7096a33f11711d8e8d88b3ec8b1b9cf9d8b550bc31a7f`。

四个官方文件均对照 Mendeley public API 的字节数与 SHA-256：

| 文件 | bytes | SHA-256 | PNG |
|---|---:|---|---:|
| metadata.csv | 316,209 | 14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527 | - |
| imgs_part_1.zip | 1,245,184,680 | 0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca | 911 |
| imgs_part_2.zip | 1,126,646,990 | e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b | 659 |
| imgs_part_3.zip | 1,220,565,093 | ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c | 728 |

三个 ZIP 的 CRC 全部通过；2,298 个 PNG 文件名并集与 metadata 的 2,298 个
`img_id` 完全相等。

## 2. metadata 与 taxonomy

- 官方记录：2,298 images、1,373 patients、1,641 lesions。
- 官方诊断计数：ACK=730、BCC=845、MEL=52、NEV=244、SCC=192、SEK=235。
- 只读取官方 `diagnostic` 字段；映射为 ACK->akiec、BCC->bcc、MEL->mel、
  NEV->nv、SCC->scc、SEK->bkl。
- PAD 不提供 df、vasc；没有补造标签或 support。

## 3. 完整参考池严格去重

构建器：`tools\prepare_pad_ufes20_external_classification_v1.py`，SHA-256
`f78311911d94bdb0cb8b5f17848cac28eb97671232219ac2f43eb50c3f45d3f8`。

实际逐图审计的参考记录：

- ISIC 2019 official training：25,331；
- HAM10000 在 ISIC 2019 中的子集别名：10,015；
- IMA++ raw：14,967；
- IMA++ binary v1 manifest：11,409；
- ISIC 2018 Task 1 external candidate：89。

对 ID、文件 SHA-256、尺寸、严格 `pHash+dHash+尺寸` 执行审计。PAD 与完整参考池
没有交叉重复排除。候选内部按严格哈希连通分量处理：

- 26 张处于冲突诊断分量，整组隔离，不多数票；
- 3 张为同诊断严格重复的非 canonical 副本，排除；
- 最终保留 2,269，排除 29。

最终保留：patients=1,371、lesions=1,630、patient-lesion groups=1,876；
akiec=723、bcc=833、bkl=234、mel=52、nv=243、scc=184。

## 4. 发布物与独立验证

正式目录：`Data\processed_pad_ufes_20_external_classification_v1`

| 发布物 | SHA-256 |
|---|---|
| manifest.csv | a465b3e473723500de0c5d6ecedf7fd1368ac2749c65b76ffdbe5c01ddc0ba37 |
| excluded.csv | 169b77ebcdb70ab1babaf6df4f0e9c1839650816c0d8f26d0161b42801ff6597 |
| duplicate_groups.csv | 0608a7af1e2b4782c324b475ed9fffc4d7bdeb5fddba62ea73f54be0c60d405b |
| group_manifest.csv | d2d70e72e90221d06acb8a26ff1ad656e8a1fce2eaa73d068d1e1b8c529ab0ed |
| hashes_sha256.csv | b76a8c4852ad946db0a5e1e05b328098cba6dd784823cace0d1f62a68ef91020 |
| audit_summary.json | 3bfcb2ad78182e5b8dd8664298834716e8ab9e439ce474c399223ad30ee98a51 |
| _SUCCESS | 1ede03c77b1e3ef12932351e943589e9ca42be803d02dcd0bdcfa40e054d73f8 |

独立验证器 `tools\validate_pad_ufes20_external_classification_v1.py`（SHA-256
`703c3007f60f9611f3c4a629310e5c974a8ea1b8c6fc2d3b8dff22021fb78160`）重新读取
2,269 张图像，逐图核对 SHA-256、尺寸、manifest、patient-lesion group、taxonomy、
final duplicate action、门禁与 `_SUCCESS`，结果 PASS。

目录只有 `external_candidate` 角色；没有 train/val/test split、mask 或 combined
training YAML。没有把 PAD 加入任何 loader 或训练配置。

## 5. 评价边界

当前禁止加载模型在此数据上运行指标。未来只有正式模型、checkpoint SHA-256、阈值、
温度和评价脚本全部冻结，且用户当次明确授权后才可评价。必须同时报告 full-8 与
supported-6 macro 指标，不在 PAD 上拟合温度/阈值，也不得根据外部结果回训。

PAD 是智能手机临床图像域，且并非所有病灶都经活检确认；结果只能解释为外部域候选
压力测试，不能替代 IMA++ internal blind test。
