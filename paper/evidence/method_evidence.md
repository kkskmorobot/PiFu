# 方法与数据证据核验

核验日期：2026年11月9日。范围为原始代码、配置、冻结数据清单、选定checkpoint清单及其元数据；未训练、未推理、未评价、未修改原始材料。原始根目录为 `E:\App\DaiMa\JetBrains\Project\PiFuV2`。本文件供主代理合并进唯一的《重写依据与终检.md》，不进入论文正文。

## 核心证据充分性

|章节／论点／图表|原始来源（相对项目根目录）|本轮核验状态|缺口与处理|
|---|---|---|---|
|第3章 LGLA定义、图3-1/3-2|tools/lgx/imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43.py，第45—227、241—289行；对应LG2、LG3实现|已回溯原始证据；三运行源文件与冻结preflight哈希一致|不能把构造函数默认hidden_channels=128当成实际分类器宽度；实际继承256|
|停止梯度、插值、池化、回退|同上第116—160行|已回溯原始证据|A=sigmoid(S.detach().float())；不是sigmoid后反传，不是真值掩膜|
|投影共享、门与分类器|同上第93—109、162—174、245—255行；冻结LGX trainer.pt的scale_channels元数据|已回溯原始证据|P3=256、P4=512；分类器256→256→8。门512→128→2|
|均衡项与批次缩放|同上第45—57、208—227行；tools/imaplusplus_partial_multitask_v1.py，第251—289行|已回溯原始证据|均值的平方而非平方均值；只在有分类标签样本上计算；追加0.5×0.01×B×R|
|基础分割损失|.venv/Lib/site-packages/ultralytics/utils/loss.py，第1275—1390行|已读取原始安装依赖实现，与checkpoint记录版本一致|原始训练fingerprint未包含依赖源码文件；没有对应git提交，辅助记录保留此复现性边界，不声称全部依赖逐字节历史一致|
|第3章 部分标签流程图|tools/imaplusplus_partial_multitask_v1.py，第71—115、180—195、260—289行|已回溯原始证据|缺分割=255忽略，缺分类target=-1；不是全背景，不是多候选标签|
|第4章 模型版本与张量|models/yolo26m-sem.pt；experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/checkpoint_bundles/epoch_0098/trainer.pt|已读取ZIP内data.pkl的pickle操作码，不执行反序列化|预训练version=8.4.50；训练version=8.4.101；git branch/commit为空，不编造提交|
|第4章 数据规模与类别图|tools/lgx/lgx_development_manifest.csv；Data/processed_imaplusplus_binary_v1/manifest.csv；Data/processed_isic2019_classification_incremental_v1/manifest.csv|已对冻结行重新计数|分割+分类覆盖存在交集，不直接相加；图表可直接取method_data.json|
|数据去重与划分|tools/prepare_imaplusplus_binary_v1.py，第365—435行；tools/prepare_isic2019_classification_incremental_v1.py，第339—473行；两个audit_summary.json|已回溯原始证据|缺患者/病灶元数据较多，不把可用字段不重叠写成完全患者独立|
|增强与共同训练条件|九个主运行原始results子目录args.yaml；各tools运行目录preregistered_config_v1.yaml|已回溯原始证据|只列既定配置；通用库支持功能不等于本次已启用|
|早停与权重选择|tools/lgx/imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43.py，第338—391行；tools/imaplusplus_partial_multitask_utility_selection_v1.py；各checkpoint_bundles/选定epoch/manifest.json|已回溯原始证据|早停适应度与最大J别名不是同一概念；按既定最大J报告，不称临床可行|
|AB1/AB2控制因素|tools/ab1/imaplusplus_partial_multitask_lgla_v1_unguided_gate_balance_seed43.py；tools/ab2/imaplusplus_partial_multitask_lgla_v1_fixed_equal_fusion_seed43.py|已回溯原始证据|AB1使global/local描述相同；AB2不调用门并关闭均衡项；不能说完全单因素隔离各贡献|

## 独立复核结果

- `verify_method_data.py`读取三个冻结CSV，重数得到联合train=23462、val=1744；分割10483/292，分类13715/1488，双监督736/36。
- 分类有效图像分组数train=5812、val=587。联合类别分布在method_data.json的development字段。
- train与val的非空isic_id/group_id/image_sha256/lesion_id/patient_id交集均为0；两集合分别存在10397/325条空lesion_id与22219/1632条空patient_id。缺失元数据是可辨识边界，不能由不重叠字段推出患者独立。
- LG1/LGX/LG2/LG3/AB1/AB2各15条锁定输入均重算SHA-256一致；Run18/Run22/Run24各17条均一致，共141条（含共享重复文件记录）。包含实际运行代码、配置、清单及预训练权重；不修改原文件。
- 九次运行选定checkpoint manifest的epoch与既有别名对应；正式可见epoch为内部0基epoch+1。其他主指标数值由结果核验代理独立负责。
- LGX冻结trainer.pt以pickletools读取操作码，记录scale_channels=(256,512)、版本8.4.101、日期2026-09-13、git提交为空；预训练权重版本8.4.50。未加载模型，未进行任何前向。

## 关键数学细节

1. `attention = interpolate(sigmoid(S.detach().float()), size, bilinear, align_corners=False).clamp(0,1)`。
2. `local = sum(F*A)/max(sum(A),1e-6)`；`sum(A)/(H*W)<=1e-4`时局部向量回退全局平均。mass_fraction不是阈值后面积比例。
3. 同尺度global/local共用一个Linear，跨尺度独立；P3投影256→128，P4投影512→128。每路拼接256维。
4. 门512→128→2，SiLU，最后Linear全零初始化。分类256→256→8，SiLU及Dropout0.20。
5. `R = mean(q_local-q_global over has_classification)^2`；均值为零不意味着平均Softmax权重严格0.5，更不意味着每个样本0.5。
6. BCE对整个批次有效像素均值；Dice对有效像素整体汇总，平滑1；辅助头只含0.4 BCE，不含辅助Dice。底层语义损失已经整体乘B。
7. 总损失为 `B [BCE + Dice + 0.4 AuxBCE + 0.5 (ClsCE + 0.01 R)]`。ClsCE为有分类标签样本均值。无有效集合对应项为0。
8. 参数算术交叉核对：Plain分类MLP为198920参数；LGLA两个投影98560、融合门65922、分类MLP67848，合计232330；增量33410与记录一致。这里仅对模块尺寸做算术检查，不作为新增模型分析或推理实验。

## 评价口径传达

本地SemanticMetrics在有效像素上累计混淆矩阵，二值任务返回前景IoU；开发Dice由2IoU/(1+IoU)得到，非逐图Dice平均。原始来源 `.venv/Lib/site-packages/ultralytics/utils/metrics.py:1706—1755`、`tools/imaplusplus_partial_multitask_v1.py:449—474`。分类按固定八类算各类F1后平均，零分母返回0；组内softmax概率算术平均再argmax。分组字段空时回退isic_id。测试集脚本的汇总规则由结果代理另核，不能以开发validator推断所有测试路径。

## 草稿与图表接口

`chapters_3_4.json`独立撰写第3章约4559字符、第4章约4794字符（不含表格及公式字符串）；非旧稿改写。JSON顶层chapters，章内sections与blocks，块类型p/equation/table/figure_placeholder。标题不含章节编号，便于主代理统一加入。

第3章12个公式，含latex、label和explanation；需转换为可编辑Word公式。图占位3-1整体、3-2 LGLA细节、3-3标签与损失；图4-1数据覆盖与分布、图4-2实验组织。草稿已预先按这些图号引出。可合并图4-1的两个子图，避免与类别表机械重复。第4章7张表给出全部数据，不要求机械保留所有表，主代理可为页面密度调整。

## 剩余边界

- 库提交号不能恢复：checkpoint未保存，保留版本和本地实现路径，不编造git SHA。
- 依赖包并非原始锁定输入的一部分：不能声称依赖实现已通过历史文件哈希同一性证明。
- 本代理未渲染Word，也未作页面验收；由主代理统一完成。
