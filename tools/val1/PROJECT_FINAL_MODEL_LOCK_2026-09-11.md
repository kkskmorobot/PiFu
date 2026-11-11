# PiFuV2 最终研究候选锁定（2026-09-11）

`FINAL_MODEL_LOCK_STATUS=PASS`
`DEVELOPMENT_PHASE=CLOSED`
`FORMAL_FEASIBLE_CHECKPOINT_COUNT=0`
`FINAL_EVALUATION_AUTHORIZED=FALSE`

## 锁定内容

- 架构：Plain YOLO26m P3/P4 partial-label 多任务模型；二值病灶语义分割头与八类图像分类头。
- 推理决策：八类原始 argmax；`calibration_temperature=1.0`。Val1 固定代价规则已拒绝。
- 输入尺寸：`640`。
- 二值语义概率阈值：`0.5`。
- taxonomy 顺序：`akiec, bcc, bkl, df, mel, nv, vasc, scc`。
- taxonomy 文件：`tools\classification_taxonomy_v1.yaml`，SHA-256 `ffd14ba2e73c8df5d3df5af67007db122d4397adbaf9ef4475fdb9858f802b8e`。
- 数据、manifest、split、mask 与 taxonomy 均保持冻结，不允许为后续结果回改。

## 冻结权重 bundle

三个 seed 的 best-Joint bundle 作为最终研究候选面板同时冻结，后续不得依据 test/external 表现从中事后挑选：

| 证据 | 可见 epoch | bundle fingerprint | trainer.pt SHA-256 | runtime_state.pt SHA-256 |
|---|---:|---|---|---|
| Run18 seed42 | 50 | `2ed64c4aa90c6d90d15ace2158de0866e570f71ad8a3cb3bbc3e2b7d0fd2b7a3` | `71a2308299fb3b7287c891bfc590f4b0b890f89750c94c081b715a0ced4fa312` | `d746270550e867926d91cd498984ac4e339a0badf953f66a2b74ccbdd7e3d7bd` |
| Run22 seed43 | 60 | `6a9cc96bd2d21f28b9c17ed745a50e76846f05d9c96153b2bad69f59c7257e82` | `be15ce40b874682d1c95246baebcd0f4d5087b5a0289699de0f009f360089535` | `0197f9300a165bd4e6f294a9739b83516ef01a9133f6123763d51112c98e978b` |
| Run24 seed44 | 78 | `c856d97b12202fac2e400ecbcd94d5ad91c705981d6c692cda827e99e28929e6` | `8ca22463f80c0e816b38972a2369793f689f8e003303c6ac83c28cb4c4ae0b62` | `56295a325b4c567e400dc5546284431675fdb0c0f96c7c7f02edb5987b4e80f4` |

三个 bundle 的 `_SUCCESS`、文件大小、SHA-256 和 manifest fingerprint 已复核通过。正式推理须沿用 Val1 已验证的精确 EMA 恢复方式，不能用压缩序列化的 fp16 EMA 替代无损 fp32 EMA。

## 解释边界

这是一项“最终研究候选锁定”，用于关闭 development 阶段和防止最终评估后的反向调参；它不意味着存在满足临床风险约束的 checkpoint。三个 seed 的正式 feasible checkpoint 数仍为 0，论文必须明确报告。

Run22 seed43 在三个 Plain seed 中具有最高 development Joint，可作为单模型描述时的预先指定代表；三 seed 统计仍必须完整报告，且不得根据未来 test/external 结果改选代表。

## 后续门禁

当前没有下一次模型训练，也没有可点击的 Run35 入口。下一步是汇总 development 结果与制作论文材料。

IMA++ test 与 ISIC2018 external 仍未授权。若未来进行一次性最终评估，必须先建立独立协议，明确三个冻结 seed 的报告方式与禁止事后选择规则，完成静态/GPU/恢复与发布门，并取得用户新的明确确认。PAD-UFES-20 继续排除。
