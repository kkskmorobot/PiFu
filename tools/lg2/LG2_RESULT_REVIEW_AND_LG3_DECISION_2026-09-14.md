# lg2 训练结果复核与 lg3 决策

日期：2026-09-14（+08:00）

## 结论

`LG2_RESULT_AUDIT=PASS`

`LG2_OFFICIAL_CSV_SCREENING_GATE=PASS`

`LG2_ATTENTION_DIAGNOSTIC_GATE=PASS`

`STANDALONE_FUSED_REEVALUATION_SCREENING_GATE=FAIL`

`NEXT_EXPERIMENT=LG3_LOCKED_REPLICATION_SEED44`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

按预注册的权威训练 CSV/selection_state 口径，lg2 相对同 seed42 的 Plain Run18 整体更好，四项核心门槛
全部通过；门控机制也正常。独立加载 checkpoint 的 standalone/fused validation 复评只得到很小的
Group Macro-F1 与 Joint 增益，未达到数值门，须作为复核限制保留。为避免针对 seed42 validation
继续调结构，决定不改模型，按原路线完成最后一个 seed44 lg3。

## 完成性与 checkpoint 审计

- lg2 完成 100/100 个可见 epoch；中途安全停止后从完整 bundle 自动恢复，最终正常完成。
- best Binary IoU alias：`epoch_0020`（可见 epoch 21）。
- best Joint 与 best Group Macro-F1 alias：`epoch_0073`（可见 epoch 74）。
- latest：`epoch_0099`（可见 epoch 100）。
- checkpoint v2 管理器重新验证上述三个 bundle 的 `_SUCCESS`、manifest、文件大小、SHA-256、
  trainer/runtime epoch、运行状态键和 fingerprint，全部通过。
- `epoch_0073` runtime selection_state 保存 Group Macro-F1=`0.62173`、Joint=`0.68868`，与 CSV
  及 alias 完全一致。
- 正式完成标记记录 test metrics=false、external metrics=false。

## 权威 best-Joint 同行比较

下表使用各自 `best_joint_score` alias 对应的训练 `results.csv` 同一行，不拼接不同 epoch 峰值。

| 指标 | Plain Run18 | lg2 | lg2 - Plain | 判读 |
|---|---:|---:|---:|---|
| Binary lesion IoU | 0.76323 | 0.76284 | -0.00039 | 基本持平，门槛内 |
| Image Macro-F1 | 0.54417 | 0.56058 | +0.01641 | 更好 |
| Group Macro-F1 | 0.61023 | 0.62173 | +0.01150 | 更好 |
| Group Macro-Recall | 0.59915 | 0.59719 | -0.00196 | 略差 |
| MEL→NV | 0.38776 | 0.39796 | +0.01020 | 更差（越低越好） |
| malignant→benign | 0.27536 | 0.27053 | -0.00483 | 更好（越低越好） |
| Group NLL | 0.75522 | 0.85321 | +0.09799 | 更差 |
| Group Brier | 0.36405 | 0.37708 | +0.01303 | 更差 |
| Group ECE | 0.06814 | 0.08326 | +0.01512 | 更差 |
| Joint | 0.68246 | 0.68868 | +0.00622 | 更好 |

四项预注册门均通过：Group Macro-F1 ≥ +0.005、Joint ≥ +0.003、Image Macro-F1 不下降、Binary
IoU ≥ -0.005。固定风险约束仍未满足，不能把该 checkpoint 称为风险可行模型。

## 门控诊断与复评限制

冻结 validation、best-Joint `epoch_0073/trainer.pt` 的独立诊断：

- local gate mean=`0.49579`、population std=`0.19644`、median=`0.48910`；
- global gate mean=`0.50421`；
- P3/P4 attention mass fraction mean=`0.30745`，fallback rate=`0`；
- 所有数值有限，local gate mean/std 门均通过；不存在 lg1 式门控坍缩；
- test、ISIC2018 external、PAD-UFES-20 均未访问。

standalone/fused 复评得到 Binary IoU `0.76288`、Image Macro-F1 `0.55349`、Group Macro-F1
`0.61155`、Joint `0.68304`；相对 Run18 的 Group Macro-F1/Joint 仅 `+0.00132/+0.00058`，低于
预注册数值门。这不改写已保存的训练期选择状态，但说明 checkpoint 独立推理复现的优势不够稳健；
最终论文和 sum2 必须同时披露，不能只报训练 CSV 峰值。

## 两个已完成成功版 seed 的方向

按权威 CSV，seed43 lgx 与 seed42 lg2 的 Group Macro-F1、Joint、Image Macro-F1 均分别优于同 seed
Plain。两 seed 平均相对 Plain：Binary IoU `+0.00724`、Image Macro-F1 `+0.01153`、Group
Macro-F1 `+0.01857`、Joint `+0.01353`；Group Recall `-0.00280`，malignant→benign
`+0.00725`。核心方向一致，但风险指标存在取舍。

## 下一次训练决策

选择按原计划继续 `lg3`，不进行结构改进：

1. 两个 seed 的权威核心方向一致，当前首要问题是第三 seed 可复现性；
2. 门控在两个 seed 均非坍缩，不存在必须先修复的结构故障；
3. 此时根据 seed42 的独立复评继续改模块会扩大 validation 适配风险；
4. lg3 完整锁定 lgx/lg2 结构与超参数，仅改 seed=44、实验标识和配对 Plain=Run24；
5. lg3 完成后停止长训练，先汇总三 seed，不自动进入消融、sum2 或 test2。
