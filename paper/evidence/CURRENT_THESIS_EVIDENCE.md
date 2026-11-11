# 当前论文权威证据链

**当前状态（2026-09-24）：第二步完整论文已完成。** Word、PDF与核查说明位于LunWen/论文重写_20260924；新的图表、参考文献及逐页排版证据位于Data/thesis_rewrite_20260924。以下第1至10节保留第一步的历史审计记录，旧“等待继续”状态以本段及第11节为准。

审计日期：2026-09-16；2026-09-20复核。状态：第一步资料汇总及当前状态复核完成；未开始新一轮论文正文写作。2026-09-16补充记录见07_current_material_audit_20260916/SUPPLEMENT_ACCEPTANCE.md，2026-09-20复核记录见08_first_step_recheck_20260920/FIRST_STEP_COMPLETION.md。下文为现有材料事实入口。全部原始路径以 E:/App/DaiMa/JetBrains/Project/PiFuV2 为根，补充快照位于 LunWen/Data/07_current_material_audit_20260916/sources；SOURCE_MANIFEST.json记录逐文件来源和SHA-256。2026-09-20复核中，480个原始来源与已有清单哈希全部一致，42个补充材料文件也全部一致；未训练、改数据或重跑测试。

## 1 当前任务、数据、输入输出

真实任务为二值病灶语义分割与八类图像级分类，采用partial-label异构监督。类别顺序akiec、bcc、bkl、df、mel、nv、vasc、scc。输入训练尺寸640、RGB图像；输出语义分割logits与8类分类logits，分类用softmax/argmax，固定评价分割阈值0.5、温度1.0。不是疾病像素标签广播，也不是目标检测框分类。

IMA++有效11409对，train/val/test=10483/292/634；ISIC2019去重增量分类14431张，train/val=12979/1452。联合train=23462，val=1744；train分割/分类/双监督=10483/13715/736，val=292/1488/36。已直接数冻结manifest，见supervision_counts.json。分类only的mask=255忽略，不充当背景监督。原始证据：Data/processed_imaplusplus_binary_v1/audit_summary.json、Data/processed_isic2019_classification_incremental_v1/audit_summary.json、tools/lgx/lgx_development_manifest.csv、tools/classification_taxonomy_v1.yaml。

IMA++固定test634张有分割标签，其中仅56张有MEL/NV分类标签（16/40）；ISIC2018增量external89张，仅分割。PAD-UFES-20虽有准备文件，但不属于当前训练或最终评估。禁止报告完整八类test Macro-F1/Joint。证据：tools/test2/inputs.json、manifest_test.csv、manifest_external.csv及正式result.json。

## 2 基线、新实验与冻结模型

基线是Plain YOLO26m P3/P4 partial-label，seed42/43/44分别Run18/22/24。成功改进结构是带门控均衡的LGLA v1，分别LG2/LGX/LG3。用户于2026-09-20确认新增训练共六次，即LG1、LGX、LG2、LG3、AB1、AB2；sum2是汇总，Test2是冻结评估，均不计为训练。

|实验|seed|实际CSV轮数|既定最佳Joint可见轮次|原别名bundle|IoU|图像Macro-F1|分组Macro-F1|Joint|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|LG1|43|100|84|epoch_0083|0.75967|0.53761|0.61057|0.68105|
|LGX|43|100|99|epoch_0098|0.77863|0.55681|0.65375|0.71346|
|LG2|42|100|74|epoch_0073|0.76284|0.56058|0.62173|0.68868|
|LG3|44|100|97|epoch_0096|0.77051|0.56096|0.6443|0.70459|
|AB1|43|98|68|epoch_0067|0.76719|0.54533|0.61436|0.68653|
|AB2|43|79|49|epoch_0048|0.76856|0.51652|0.59544|0.67648|
|18|42|100|50|epoch_0049|0.76323|0.54417|0.61023|0.68246|
|22|43|93|60|epoch_0059|0.76377|0.55017|0.62811|0.69263|
|24|44|100|78|epoch_0077|0.76393|0.54301|0.60641|0.68063|

表中每行指标来自原来保存的best_joint_score别名所指同一CSV行；没有拼接各指标峰值。CSV可见epoch比bundle零基epoch大1。精确路径、实际参数与完整行见run_evidence.json、run_summary.csv。best.pt/last.pt只登记，不替换已冻结选择。

LG1：无均衡正则的病灶引导全局/局部分支，seed43首轮；LGX：增加分类标签样本批次均值门控logit平衡正则，权重0.01；LG2/LG3：成功结构的seed42/44复现；AB1：以全图常量空间权重替代病灶引导，保留参数图；AB2：0.5/0.5固定融合，保留门模块参数但不调用且无门控正则。两项结构对照仅seed43，不宣称三seed消融。

冻结模型包：LG2 epoch_0073、LGX epoch_0098、LG3 epoch_0096。Test2对应可见74/99/97轮；seed43为预先指定展示代表。原始alias、bundle manifest及_SUCCESS已补齐；9个run共18个trainer/runtime_state二进制文件SHA-256与bundle清单一致。大权重留在原位，见checkpoint_inventory.json。

## 3 结构、训练与比较条件

源码tools/lgx/imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43.py：P3/P4输入层[16,13]，主分割logit先detach再sigmoid得到病灶概率；局部概率加权池化与全局平均池化、每尺度共享128维投影、拼接、两路softmax门融合、MLP分类。空/极小注意力回退全局池化。仅替换分类头，语义分支未改，不把真值mask输入分类器。分类标签子集上增加批次平均(local-global)门logit平方正则，权重0.01。

args/config记录输入640、batch16、AdamW、lr0=0.001、lrf=0.01、weight_decay=0.0005、上限100轮，延迟效用早停从40轮、耐心30。六个新实验及三个基线的实际公共参数单列training_conditions.json；开发manifest哈希与划分见split_manifest_audit.json。不以相同seed代表统计显著性，且各run实际停止轮次不同。

实际增强：HSV=0.005/0.10/0.10、旋转5度、平移0.03、缩放0.10、上下/左右翻转0.5；mosaic/mixup/cutmix/copy_paste关闭。只按args及原配置写，不补不存在的增强或去毛发。旧YOLO实例分割格式转换位于tools/run1/prepare_skin7_dataset.py（mask轮廓、approxPolyDP、归一化坐标），run2/run3调用演进版本；当前主线读取二值语义mask与多任务manifest，不能把历史polygon当作当前分割监督。已收录历史转换源代码与一份实际label样例。

## 4 已证实的改进、失败与指标

从原CSV配对复算：LGLA相对Plain的三seed平均图像Macro-F1 +0.01367，分组Macro-F1 +0.02501，Joint +0.01700，三个seed均增加；IoU平均+0.00702，仅2/3 seed增加。这些为绝对指标差，不是相对百分比。Joint=sqrt(Binary IoU × Group Macro-F1)，不得与mAP混排。证据：paired_recomputed.json及原CSV，既有06_lgla_extension/01_development/paired_development.csv。

参数量Plain 14507338，LGLA 14540748，增加33410（约0.2303%）。来源tools/val1/RUN34_PREFLIGHT_REPORT.json、tools/lgx/LGX_PREFLIGHT_REPORT.json；AB1/AB2记录同参数量，不等同同有效计算量。统一单机性能材料已补齐：Run22/LGX直接前向均值5.4703/5.6642 ms，均值换算吞吐182.8042/176.5487 FPS；同图预测管线6.2254/6.5456 ms。现有工具输出THOP 59.574067/59.679130 GFLOPs，仅作为Ultralytics步长面积外推估计，不能称作完整模型精确FLOPs。详细方法限制见SUPPLEMENT_ACCEPTANCE.md。预检硬件记录RTX5080、torch2.8.0+cu129、CUDA12.9，见LGX预检报告；当前安装版本不能反推所有历史训练环境。

LG1最佳Joint0.68105未超过Plain seed43的0.69263，门控塌缩诊断留存；不能省略失败轮。AB1/AB2 Joint分别0.68653/0.67648，小于完整LGX0.71346。历史ECA、Spatial7、CBAM、风险训练结果保留在旧材料包，专项变化与总体改善分开，尚未在本次逐一重审所有历史run。开发集风险/校准没有形成稳定优势，feasible不能写临床合格。

## 5 最终评估证据与边界

权威结果：experiments/test1/results/result.json与experiments/test2/results/result.json；Test2 release.md记录完成和一次性授权已消费。Test2三seed内部IoU均值约0.78558，Plain约0.78547；受限分类Accuracy约0.79167对0.77976，MEL Recall约0.43750对0.45833；external IoU约0.59449对0.63347。描述为内部分割基本持平、受限准确率略高，不能写全面优于或临床有效。

Test1和Test2内部/外部manifest哈希相同。LGLA扩展发生在已查看Test1后；扩展计划明确允许相同测试集上的补充评估。因此即便Test2之前锁定了权重，也不是全新独立盲测。未重跑评估，未利用结果开发模型。此限制必须保留，不把“没有样本重叠”与“未经历测试反馈”混为一谈。

## 6 当前系统功能与未证实事项

权威源码QianDuan/app.py、index.html、启动脚本已快照。静态确认：单图上传/样例读取、模型发现切换和单模型缓存、阈值控制、分类概率、mask/轮廓/包围矩形显示、JSON报告导出、复核JSONL落盘、100张本地抽样批量异步任务、进度及详情展示、输入异常分支。服务使用Python ThreadingHTTPServer；BATCH_JOBS存内存，复核记录文件为runtime/annotations.jsonl。无关系数据库证据。

批量从Data/processed_skin7_v3/images/{test,val}及旧metadata读取，属于界面演示样本，与正式多任务开发/测试split不同；未见任意用户多图上传接口，不写“多图上传已实现”。已收到当前版本真实服务测试：23个接口用例记录PASS，8个浏览器用例记录PASS，代码哈希与当前源码一致；原JSON、脚本和JSONL摘录保存在supplements_20260916/system_test。记录的测试模型为LGX/Run22，不改称LG3。批量零失败有UI记录；接口脚本未正确断言failed计数，按检查范围使用。导出项仅证实界面触发及成功提示，未核对下载文件落盘。旧纯函数记录不充当新版端到端测试。本次仅复核已有测试，不重跑。

实际前端展示使用LG3 seed44 best.pt。经只读张量比对，其参与推理的428个状态张量均与冻结epoch_0096 EMA按部署FP16精度转换后的值一致，LG3权重对应缺口关闭。该结论是checkpoint来源与推理权重对应，不是FP32逐位同一或输出等价性试验。仅训练辅助头的BN方差存在FP16范围截断，不参与eval前向；细节保留内部verification记录，不写成前端模型错配。论文按LG3展示、实际阈值描述，不要求所有历史Run一致；模型统计仍保留三seed，已有LGX性能与软件测试不改名为LG3。

截图中的10.0 mm为HTML静态示意标尺，无标定证据，不能声称真实物理尺寸。风险等级为类别规则映射，不能称为独立医学风险预测模型。

## 7 数字与插图来源索引

- 数据规模/划分：本包supervision_counts.json、split_manifest_audit.json及sources/Data的审计和manifest。
- 全部本次训练数字：run_evidence.json、run_summary.csv以及sources/experiments每轮results.csv；配对差值paired_recomputed.json。
- 最终测试：sources/experiments/test1与test2/results；原06目录的test2_vs_test1.csv作便利表。
- 参数/设备：sources/tools/lgx、ab1、ab2和val1预检报告；速度及工具估算FLOPs见supplements_20260916/performance/unified_performance_results.json。
- 方法结构与预处理：sources/tools源代码；结构图第二步据此自行绘制并保存源文件。
- 系统功能：sources/QianDuan；UML及数据对象图第二步据当前实现绘制，不沿用旧关系数据库假设。
- 现有实验图：06_lgla_extension/01_development及04_repro；图与生成脚本必须配对引用。新的论文图尚未制作。
- 四张现有截图：Data根目录单张预测示意图1/2.png、批量预测示意图1/2.png，详见SCREENSHOT_AUDIT.md；只支持截图所示演示状态，不替代系统测试或正式评估。
- 模板：TEMPLATE_REQUIREMENTS.md、template_comments.json、template_formatting.json及完整模板包；已有Microsoft Word只读导出PDF及29张PNG和完整逐页检查报告，已核对29页文件、32条批注和模板源哈希不变；本轮独立抽查1、6、22、29页。此项关闭，不称LibreOffice兼容性验证。

## 8 材料完整性与下一步停止点

旧两个manifest共270条记录全部符合记录哈希；其中7条Val1原路径不存在，但Data副本完整。许多06记录source指向包自身，单独哈希只能证明复制件完整，不能证明独立来源；本次已补充原始tools/experiments证据并独立复算主指标。SOURCE_MANIFEST.json记录新补充文件；collection_verification.json与PACKAGE_CHECKSUMS.json记录检查范围。

新版系统测试、统一性能材料、模板重新渲染记录均已收录；LG3部署权重已核对，相关缺口关闭。训练数量按用户确认的六次新增训练和六个可追溯训练目录列举。面向本科毕业论文，现有材料已经足够支撑模型对比、系统设计、实现和测试章节；FLOPs只在确有需要时标注为工具估算，也可以不作为核心结论，逐次计时数组和额外统计检验不作为开写前硬性缺口。材料汇总完成，正文等待用户“继续”。

## 9 本轮确认的写作组织方式

按方法设计、三seed复现和结构对照统一组织LG/AB实验，不要求在正文叙述内部冻结、授权及恢复的工程历史，也不虚构同时训练的时间线。方法优劣照原始指标量化。涉及Test2时简洁说明其为既有固定测试集上的补充比较，不能称为全新的独立盲测。LG3作为系统展示实例，与LGX工程基准/测试记录分别准确标注。

## 10 本轮资料复用核对入口

2026-09-20按用户“先完成第一步、允许复用前面模型资料、新增训练共六次”的要求，完成材料复用核对。简明清单见`08_first_step_recheck_20260920/本轮资料清单.md`，本轮机器核对记录见同目录`CURRENT_REUSE_CHECK.json`。480条原始来源及副本、42项补充材料均与归档一致，九个训练记录的既定权重对应CSV行及Test1/Test2比较均值复核一致。模板32条批注和49个归档内部文件也与原DOCX一致。现有资料足以进入本科应用开发型论文写作，开始前无须用户补齐其他资料；本轮未撰写正文，等待用户“继续”。


## 11 第二步论文完成记录（2026-09-24）

已从学校原模板副本生成完整本科应用开发型论文，42页，七章正文，16幅图、18张编号表、6组公式、16条参考文献。个人信息取自开题报告，新增训练按六次处理。480条来源与42项补充材料复用核对结果见thesis_rewrite_20260924/CURRENT_REUSE_CHECK.json。

图表数据来源索引为thesis_rewrite_20260924/evidence_sources.json，表格数据为tables.json，绘图源为make_figures.py及figures内矢量PDF，参考文献核对为REFERENCE_CHECK.md。软件记录采用原2026-09-16测试，不声称本轮重跑。新增论文叙述与原实验结果相符：LGLA验证集分类及综合指标改善，内部分割基本持平，外部分割和MEL召回不足。

模板结构检查及42页逐页核查已完成，详见论文重写_20260924/论文核查说明.md和Data/thesis_rewrite_20260924/FINAL_STRUCTURAL_AUDIT.json。未修改训练、测试划分、模型源码和权重，也未覆盖其他模型的论文。
