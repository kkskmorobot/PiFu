# lg3 训练结果复核与 ab1 决策

日期：2026-09-15（+08:00）

`LG3_RESULT_AUDIT=PASS`

`LG3_OFFICIAL_CSV_SCREENING_GATE=PASS`

`LG3_STANDALONE_VALIDATION_SCREENING_GATE=PASS`

`LG3_ATTENTION_DIAGNOSTIC_GATE=PASS`

`THREE_SEED_EFFECTIVENESS_GATE=PASS`

`NEXT_EXPERIMENT=AB1_LESION_GUIDANCE_ABLATION_SEED43`

`TEST_METRICS_USED=FALSE`

`EXTERNAL_METRICS_USED=FALSE`

## lg3 完成性与 checkpoint 审计

- 完成 100/100 个可见 epoch，完成标记 last internal epoch=`99`；
- best Binary IoU=`epoch_0043`（可见 44）；
- best Joint 与 Group Macro-F1=`epoch_0096`（可见 97）；latest=`epoch_0099`；
- checkpoint 管理器重新验证以上 bundle 的 `_SUCCESS`、manifest、大小、SHA-256、epoch 与 fingerprint；
- best-Joint trainer SHA-256=`cb6cf3a9d40d180a9ace51da6bb2420411d95da408f62f3009e6b33a289675e2`；
- best-Joint runtime SHA-256=`b7bb2c5d926f899c66d9ee74ef5b4cac8266c763413ac3475e7a162bbba269d7`；
- fingerprint=`4a7c29f336b7ea28d8fbfe38e5e51cdff21f68ba418c78fddedbceb6956642aa`；
- selection_state 的 Group Macro-F1=`0.64430`、Joint=`0.70459`，与 CSV/alias 一致。

## lg3 相对同 seed Plain Run24

| 指标 | Plain Run24 | lg3 | 差值 | 判读 |
|---|---:|---:|---:|---|
| Binary IoU | 0.76393 | 0.77051 | +0.00658 | 更好 |
| Image Macro-F1 | 0.54301 | 0.56096 | +0.01795 | 更好 |
| Group Macro-F1 | 0.60641 | 0.64430 | +0.03789 | 更好 |
| Group Recall | 0.62047 | 0.62919 | +0.00872 | 更好 |
| MEL→NV | 0.34694 | 0.36735 | +0.02041 | 更差 |
| malignant→benign | 0.25121 | 0.26570 | +0.01449 | 更差 |
| NLL | 0.82168 | 0.96088 | +0.13920 | 更差 |
| Brier | 0.36650 | 0.38732 | +0.02082 | 更差 |
| ECE | 0.09122 | 0.11425 | +0.02303 | 更差 |
| Joint | 0.68063 | 0.70459 | +0.02396 | 更好 |

四项预注册核心门全部通过。固定风险约束仍未满足，不能称为风险可行模型。

## 独立冻结 validation 与门控诊断

独立从 best-Joint checkpoint 加载并 fuse 后复评：Binary IoU=`0.77059`、Image Macro-F1=`0.55795`、
Group Macro-F1=`0.64430`、Joint=`0.70462`。相对 Run24 四项筛选门仍全部通过。

局部门均值=`0.50788`、population SD=`0.17458`，注意力质量均值=`0.30140`，fallback rate=`0`；
全部有限，无门控坍缩。数据范围严格为冻结 validation，没有访问 test、external 或 PAD-UFES-20。

## 三 seed 判定（用于阶段门，不是 sum2 正式汇总）

权威 CSV 的 LGLA 三 seed 均值：Binary IoU=`0.77066`、Image Macro-F1=`0.55945`、Group
Macro-F1=`0.63993`、Group Recall=`0.62103`、Joint=`0.70224`。相对 Plain 三 seed 均值分别为
`+0.00702/+0.01367/+0.02501/+0.00104/+0.01700`。三组 seed 的 Image Macro-F1、Group
Macro-F1 与 Joint 都同方向提高，满足扩展计划阶段 C1 的总体有效门。

代价是 MEL→NV 均值从 `0.38095` 升至 `0.38436`，malignant→benign 从 `0.27053` 升至
`0.28019`，NLL/Brier/ECE 也变差；因此结论是核心分类与 Joint 总体更好，但风险和校准并未全面改善。

三 seed 独立 checkpoint 复评的 Group Macro-F1/Joint 均值约为 `0.63654/0.70037`，仍高于
Plain 的 `0.61492/0.68524`；seed42 的独立增益较小，需在最终 sum2 中保留该限制。

## 下一次训练决策

不再改进完整 LGLA 主结构，按原计划进入阶段 D 的优先消融 ab1：

1. 主实验已跨三个 seed 通过，继续针对 validation 调结构会扩大适配风险；
2. ab1 只把预测病灶概率权重替换为全图常数 1，其他实现和超参数复用 lgx seed43；
3. ab1 用于判断收益是否来自病灶空间先验，不是新的候选主结构；
4. ab1 完成前不启动 sum2、论文材料汇总或 test2。
