# Run10 外部域与盲测边界协议

更新时间：2026-08-04  
适用项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`

`EXTERNAL_EVALUATION_BOUNDARY_PROTOCOL_GATE=PASS`  
`EXTERNAL_SEGMENTATION_CANDIDATE_ROLE_GATE=PASS`  
`EXTERNAL_CLASSIFICATION_DATA_GATE=BLOCKED_PENDING_SOURCE_AND_STRICT_DEDUP`  
`BLIND_EVALUATION_EXECUTION_GATE=BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION`  
`TEST_METRICS_USED=FALSE`

本协议只锁定数据角色和防泄漏边界，不授权执行任何 test/external evaluation，
不创建或启动 `experiments\10_...`。

## 1. development 数据

模型训练、early stopping、超参数决定和 checkpoint 选择只能使用：

- IMA++ binary v1 train=10,483、val=292；
- 严格去重的 ISIC 2019 classification incremental v1 train=12,979、val=1,452；
- IMA++ 中继承原 split 的官方分类重叠：train=736、val=36。

不得根据任何 test 或 external 结果调整阈值、taxonomy、loss 权重、采样、增强、
训练预算、early stopping 或候选 checkpoint 排序。

## 2. IMA++ internal blind test

`Data\processed_imaplusplus_binary_v1` 的 test=634 是项目最终 internal blind test。

- 正式模型和全部决策冻结前禁止加载；
- 必须同时满足 evaluator 的 `--split test` 与 `--allow-test`；
- 仍需用户在当次执行前明确授权；
- 只允许一次预注册主评价，任何重复运行或事后子集分析必须单独标记 exploratory；
- test 结果不得回流训练或替换已冻结 checkpoint。

## 3. ISIC 2018 Task 1 external segmentation candidate

`Data\processed_isic2018_task1_incremental_v1` 的 89 对严格去重图像/mask 只承担
external segmentation candidate 角色。

- 永不进入 train、val、calibration、threshold tuning 或模型选择；
- 当前不得运行评价；
- 只有模型、阈值和 checkpoint SHA-256 冻结，且用户明确授权后才可解封；
- 解封后报告 binary IoU、Dice、pixel accuracy、逐图分布和 bootstrap CI；
- 结果必须标记为 external candidate，不得与 IMA++ internal test 混为一个集合；
- 若后续审计发现 patient/lesion/source 依赖或元数据不足，降级为 exploratory，
  不能宣传为独立外部验证。

## 4. external classification 数据要求

当前没有已验收的 external classification 数据，因此
`EXTERNAL_CLASSIFICATION_DATA_GATE` 保持阻塞。候选数据必须同时满足：

1. 标签来自可追溯的官方诊断字段，只映射到锁定的八类 taxonomy；不能从 mask、
   文件夹名猜测或把未知诊断强塞进八类。
2. 对 ISIC 2019 training、HAM10000、IMA++ 原始 14,967 ID、IMA++ binary v1、
   ISIC 2018 Task 1 候选执行 ID、SHA-256、尺寸与严格 `pHash+dHash` 去重。
3. 有 patient/lesion ID 时按组审计；缺失时明确报告，不能把缺失当作独立性证据。
4. 不用于温度缩放、阈值选择、类别权重、loss、采样或 checkpoint 排序。
5. 若只覆盖八类的子集，同时报告 full-8 macro 指标和 supported-class macro 指标，
   不补造缺失类别 support。
6. 数据来源、许可、下载哈希、manifest、排除清单和 `_SUCCESS` 必须独立验收。

## 5. 解封顺序

1. 仅使用 development val 完成模型选择；
2. 冻结模型结构、配置、checkpoint SHA-256、阈值和评价脚本 SHA-256；
3. 写出最新文档并由用户明确授权；
4. external candidate 与 internal blind test 分开执行、分开报告；
5. 不根据结果回训。若需要新假设，必须创建后续新编号实验并重新预注册。

