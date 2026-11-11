# lgx 训练结果复核与 lg2 决策

日期：2026-09-13（+08:00）

## 结论

`LGX_RESULT_AUDIT=PASS`

`LGX_SCREENING_GATE=PASS`

`LGX_ATTENTION_DIAGNOSTIC_GATE=PASS`

`NEXT_EXPERIMENT=LG2_LOCKED_LGX_REPLICATION_SEED42`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

lgx 相对相同 seed43 的 Plain Run22 在核心综合结果上明显更好，并且解决了 lg1 的局部门控坍缩。
按扩展计划不再修改结构；下一次训练固定为成功版 lgx 的精确 seed42 复现 lg2，并与 Plain Run18
配对。此结论仍属于冻结 validation 开发结论，不是测试集或外部泛化结论。

## 完成性与 checkpoint 审计

- lgx 从 2026-09-13 14:21:12 训练至 20:28:35，完成 100/100 个 epoch；最后完成 bundle 为
  `epoch_0099`。
- best Binary IoU alias 指向 `epoch_0030`（CSV 可见 epoch 31）。
- best Joint 与 best Group Macro-F1 alias 均指向 `epoch_0098`（CSV 可见 epoch 99）。
- checkpoint v2 管理器重新验证 `epoch_0030`、`epoch_0098` 与 auto/latest `epoch_0099`：`_SUCCESS`、
  manifest、文件大小、SHA-256、trainer/runtime epoch、运行状态键与 fingerprint 全部通过。
- 锁定 fingerprint：`3a9aa4156828bd2e5af1b1f44a62929041cde5895d01ed995c8c5af823186ac0`。
- 正式训练完成标记明确记录 test metrics=false、external metrics=false。

## 权威 best-Joint 同行比较

以下全部来自各自 `best_joint_score` alias 对应的 `results.csv` 同一行，不拼接不同 epoch 的峰值。

| 指标 | Plain Run22 | lgx | lgx - Plain | 判读 |
|---|---:|---:|---:|---|
| Binary lesion IoU | 0.76377 | 0.77863 | +0.01486 | 更好 |
| Image Macro-F1 | 0.55017 | 0.55681 | +0.00664 | 更好 |
| Group Macro-F1 | 0.62811 | 0.65375 | +0.02564 | 明显更好 |
| Group Macro-Recall | 0.64033 | 0.63670 | -0.00363 | 略差 |
| MEL→NV | 0.40816 | 0.38776 | -0.02040 | 更好（越低越好） |
| malignant→benign | 0.28502 | 0.30435 | +0.01933 | 更差（越低越好） |
| Joint | 0.69263 | 0.71346 | +0.02083 | 明显更好 |

四项预注册数值门槛全部通过：Group Macro-F1 ≥ +0.005、Joint ≥ +0.003、Image Macro-F1 不下降、
Binary IoU ≥ -0.005。需要保留的风险信号是 malignant→benign 增加约 1.93 个百分点，以及 Group
Macro-Recall 下降约 0.36 个百分点。

## 冻结 validation 门控诊断

独立诊断使用 best-Joint `epoch_0098/trainer.pt`，覆盖全部 1,746 次前向与 1,488 条官方分类行：

- local gate：均值 0.51507、总体标准差 0.19879、中位数 0.52556、P05/P95=0.18935/0.82589；
- global gate：均值 0.48493、总体标准差 0.19879；
- P3/P4 注意力质量占比均值 0.30940，tiny-attention fallback rate 均为 0.002291；
- global/local descriptor cosine 均值 0.46143，relative-L2 均值 1.71810；
- 所有数值有限，local gate mean 位于 [0.05,0.95] 且 std≥0.001；诊断门 PASS；
- test、ISIC2018 external 与 PAD-UFES-20 均未访问。

诊断脚本重新前向计算的 Image Macro-F1 为 0.55731，与训练 CSV 的 0.55681 有轻微实现/聚合粒度差异；
正式实验判定始终采用锁定 CSV 同行的 0.55681，而这两个数值都不会改变筛选门结论。

## 下一次训练决策

选择“按计划接着训练”，而不是继续改进后再训练：

1. lgx 已通过数值门与机制诊断门，继续针对同一 validation 改结构会增加适配偏差；
2. 当前最重要的不确定性已从“门控是否工作”变成“收益能否跨 seed 复现”；
3. lg2 完整锁定 lgx 的 P3/P4 LGLA、gate-balance=0.01、损失、数据、增强、优化器、早停、选择和
   checkpoint 合同，仅改 seed=42、实验标识及配对 Plain 基线为 Run18；
4. lg2 完成后先报告与 Run18 的完整配对差值，不自动继续 seed44 或消融。

## 证据位置

- 训练结果：`experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43`
- 门控诊断：上述目录下 `analysis/val_attention_diagnostics_v1/LGX_VAL_ATTENTION_DIAGNOSTICS.json`
- lg2 协议：`tools/lg2/LG2_PROTOCOL_2026-09-13.md`
- lg2 正式入口：`tools/lg2/train_lg2_from_pycharm.py`
