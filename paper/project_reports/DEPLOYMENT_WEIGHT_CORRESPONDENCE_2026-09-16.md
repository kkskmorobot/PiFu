# 部署权重对应记录

- 审计日期：2026-09-16
- 项目：PiFuV2 皮肤病变识别与分割系统
- 审计对象：前端 `QianDuan/app.py` 实际加载的 `experiments/**/results/**/weights/best.pt` 与正式冻结测试权重
- 审计结论：**条件通过（最终代表模型 LGX 对应成立；并非所有前端模型都成立）**

## 1. 结论摘要

当网页模型选择为 **LGX** 时，前端加载的 `best.pt` 可以追溯到 Test2 冻结评测中预先指定的代表模型 **LGX / seed 43 / visible epoch 99（zero-based epoch 98）**。部署文件中的 436 个状态张量与该冻结检查点 `trainer.pt` 的 EMA 快照逐张量完全一致；其推理有效参数与正式测试使用的无损 FP32 EMA 状态相对应。因此可以表述为：

> 前端选择 LGX 时，加载的是冻结 Test2 代表检查点的 FP16 部署快照；其检查点身份和推理有效参数与冻结测试权重对应。

需要同时保留两个限定：

1. 正式 Test2 使用 `runtime_state.pt::training_state.ema` 中的无损 FP32 EMA，网页 `best.pt` 是 FP16 部署文件，所以二者不是同一个二进制文件，也不应声称逐字节相同或 FP32/FP16 输出绝对位级相同。
2. 前端按 `best.pt` 修改时间倒序排列并把第一项作为默认模型；审计时默认项为 **AB2**，不是 LGX。论文演示或截图若需要对应冻结 Test2 结果，必须在页面中明确选择 **LGX**。

## 2. LGX 对应证据

### 前端实际加载文件

`experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/weights/best.pt`

- 文件大小：29,346,699 bytes
- SHA-256：`5a8e6fa655ce39a7a79af05ff44199d6e273d9ff9034b1e362c2fd1ae63e9bd9`
- 前端加载入口：`QianDuan/app.py` 的 `load_model()`，使用 `YOLO(str(model["weights"]), task="semantic")`

### 冻结 Test2 检查点

`experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/checkpoint_bundles/epoch_0098`

- Bundle fingerprint：`3a9aa4156828bd2e5af1b1f44a62929041cde5895d01ed995c8c5af823186ac0`
- `trainer.pt` SHA-256：`7db5600fcafbc5853902f234d5ee1e08c9ec44d9160d8077a5cd5af16a1e14ba`
- `runtime_state.pt` SHA-256：`6ad29b0d5422d393e79efa6eab414dabc0f731055654bf82c75e66787c0f9b33`
- Test2 固定代表：seed 43，LGX，visible epoch 99
- 正式测试状态源：`runtime_state.pt::training_state.ema`

### 张量级核对

| 核对项 | 结果 |
|---|---:|
| 状态键数量 | 436 对 436 |
| 缺失键 / 多余键 | 0 / 0 |
| 形状不一致 | 0 |
| 比较标量总数 | 14,570,130 |
| `best.pt` 对 bundle `trainer.pt::ema` | **436/436 张量完全一致** |
| `best.pt` 对无损 FP32 EMA（按部署精度转换） | 仅 1 个非推理缓冲区例外 |

唯一例外是 `model.17.aux_head.0.bn.running_var`。该缓冲区部分 FP32 数值超出 FP16 正常最大值，部署文件中被限制为 65504；但 `SemanticSegment.forward()` 只在 `self.training` 为真时调用 `aux_head`，模型评估/网页推理路径直接返回主头 `logits`，因此该缓冲区不参与前端推理。其余推理有效状态与冻结 EMA 的 FP16 部署表示一致。

## 3. 全部正式冻结候选的交叉检查

以下核对覆盖 Test1 的 Plain 三种子和 Test2 的 LGLA-v1 三种子。通过表示前端 `best.pt` 与对应 bundle 的 `trainer.pt::ema` 完整一致；不通过表示它们来自同一实验结构，但不是正式冻结测试所选的那个权重快照。

| 冻结测试 | 前端 ID | 冻结 epoch（zero-based） | `best.pt` SHA-256 | 张量核对 | 结论 |
|---|---|---:|---|---:|---|
| Test1 | Run18 | 49 | `45c06d44c89ac7297def73290b442304c527be21a1532e8b0e83c7594895ed61` | 358/428 不同 | 不通过 |
| Test1 | Run22 | 59 | `a20dd6a0349ebe2fcef85a715e555c60121d45bc281580dac6d0b933a83775ad` | 356/428 不同 | 不通过 |
| Test1 | Run24 | 77 | `20e745760c516adf734ff34a5bfe29bfcb85c541ecd40c0b3c64d3cd2828f3d1` | 428/428 一致 | 通过 |
| Test2 | LG2 | 73 | `e4237a569b475bd0f61dcc3c498671d7d19fce358a6099fd1f35804530d887c0` | 366/436 不同 | 不通过 |
| Test2 | LGX | 98 | `5a8e6fa655ce39a7a79af05ff44199d6e273d9ff9034b1e362c2fd1ae63e9bd9` | 436/436 一致 | **通过，最终代表** |
| Test2 | LG3 | 96 | `69711bfa8ed435c23e6aa7f16e547855199cf402d1ace8a0811d2731e5a08d21` | 436/436 一致 | 通过 |

因此，不能使用“前端中的所有 `best.pt` 都与冻结测试权重一致”这一笼统说法。Run18、Run22、LG2 的网页权重不得直接引用各自正式冻结测试快照的指标；AB1、AB2 等消融模型也不属于 Test2 三个冻结候选。

## 4. 审计依据与方法

1. 读取 `QianDuan/app.py`，确认模型发现规则、默认模型规则和实际 `YOLO` 加载路径。
2. 读取 `tools/test1/test1.yaml`、`tools/test2/test2.yaml` 及相应 protocol，锁定正式测试 bundle、epoch、文件大小、SHA-256 和代表种子。
3. 重新计算前端 `best.pt`、bundle `trainer.pt`、`runtime_state.pt` 的 SHA-256。
4. 在 CPU 上反序列化权重，比较状态键、形状、张量数量及逐元素值；同时核查正式测试的 `load_exact_ema()` 确实从 `runtime_state.pt::training_state.ema` 加载无损 EMA。
5. 检查 `SemanticSegment.forward()` 的评估分支，确认唯一超出 FP16 范围的辅助头缓冲区不参与网页推理。

## 5. 可用于论文/答辩的准确表述

> 系统前端支持选择多个实验模型。论文最终代表模型选用 LGX（seed 43），其前端 `best.pt` 为冻结 Test2 epoch 98 无损 EMA 权重生成的 FP16 部署快照。经 SHA-256、检查点元数据及 436 个状态张量核对，检查点来源一致；正式测试仍以冻结 bundle 中的 FP32 EMA 为准。

本记录只证明“LGX 前端部署权重与冻结代表检查点的对应关系”，不把 Test2 指标扩展到默认 AB2 或其他未通过核对的前端权重。
