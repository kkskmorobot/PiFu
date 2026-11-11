# YOLO26m development 汇总

`SUM1_STATUS=COMPLETE`
`TEST_METRICS_USED=FALSE`
`EXTERNAL_METRICS_USED=FALSE`
`PAD_UFES_20_USED=FALSE`

## 主结论

三 seed 的统一 best-Joint 比较支持 Plain YOLO26m P3/P4 partial-label 多任务架构作为最终研究候选。Plain 的平均 Joint 为 `0.68524`，ECA 为 `0.68080`，ECA+Spatial7 为 `0.68096`。三组正式 feasible checkpoint 数均为 0。

ECA 与 ECA+Spatial7 没有形成稳定的总体效用增益。标准 CBAM、BLG-MDA 演进以及 Run29–Run32 风险训练分支均保留为探索性或失败消融证据，不进入最终架构。

Val1 的固定代价规则降低了两项恶性误判，但平均 Group Macro-F1 与 Joint 的损害超过预注册保护线，因此最终决策规则保持八类 argmax。

## 三 seed 主实验均值 ± 样本标准差

| 架构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Group Recall | MEL→NV | malignant→benign | Joint | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain | 0.76364 ± 0.00037 | 0.54578 ± 0.00384 | 0.61492 ± 0.01158 | 0.61998 ± 0.02059 | 0.38095 ± 0.03117 | 0.27053 ± 0.01741 | 0.68524 ± 0.00647 | 0/3 |
| ECA | 0.76684 ± 0.00660 | 0.53850 ± 0.01146 | 0.60451 ± 0.01816 | 0.58313 ± 0.01997 | 0.42177 ± 0.02357 | 0.29469 ± 0.00967 | 0.68080 ± 0.01086 | 0/3 |
| ECA+Spatial7 | 0.76492 ± 0.00548 | 0.54380 ± 0.01486 | 0.60677 ± 0.04393 | 0.59669 ± 0.05579 | 0.36054 ± 0.03280 | 0.27536 ± 0.02556 | 0.68096 ± 0.02491 | 0/3 |


## 参数量

| 架构 | 参数量 | 相对 Plain 增量 | 相对增幅 |
|---|---:|---:|---:|
| Plain | 14,507,338 | +0 | 0.0000% |
| ECA | 14,507,348 | +10 | 0.0001% |
| ECA+Spatial7 | 14,507,544 | +206 | 0.0014% |
| CBAM | 14,548,494 | +41,156 | 0.2837% |


语义主干在现有验证输出中报告为约 `59.1 GFLOPs`；自定义分类/注意力头的总 GFLOPs 没有在所有 run 中以同一统计口径保存，因此不据此编造跨结构 GFLOPs 差值。

## 论文解释边界

- YOLO26s 只承担早期低成本筛选；YOLO26m 是正式主模型。
- Run18/22/24、Run21/23/25、Run27/28/26 构成三 seed 注意力主消融。
- Run33 是标准 CBAM 单 seed 对照；Run16/19/20 是 BLG-MDA 演进证据；Run29–Run32 是未进入复现阶段的风险分支。
- Val1 是冻结模型的 development-only 决策实验，历史预注册名为 Run34。
- 最终研究候选不等于临床风险合格模型；正式 feasible checkpoint 为 0。
- 本汇总未读取 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
