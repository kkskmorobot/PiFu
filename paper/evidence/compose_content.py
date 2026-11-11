from pathlib import Path
import json,statistics
O=Path(r'E:\App\DaiMa\JetBrains\Project\PiFuV2\LunWen\astra-6-ultra\Data\academic_rewrite');D=json.loads((O/'results_data.json').read_text(encoding='utf-8'))
def p(s):return {'type':'p','text':s}
def ps(s):return [p(x.strip()) for x in s.strip().split('\n') if x.strip()]
def fig(id,title,name,note=''):return {'type':'figure','id':id,'title':title,'file':'figures/'+name+'.png','source':note}
def tab(id,title,headers,rows,note=''):return {'type':'table','id':id,'title':title,'headers':headers,'rows':rows,'source':note}
def fmt(mean,sd=None,two=False):return f'{mean:.5f}'+(('\n' if two else '')+f'±{sd:.5f}' if sd is not None else '')
ch={'number':5,'title':'实验验证','sections':[]}
def sec(title,blocks):ch['sections'].append({'title':title,'blocks':blocks})
b=ps('''为检验病灶引导融合对联合任务的作用，本节在相同开发验证集上比较Plain与完整LGLA。两种方法均采用随机种子42、43和44，分别读取既定综合指标对应权重的同轮结果，再计算三次运行的均值和样本标准差。图像分类评价使用具有正式分类标签的样本，分割评价使用具有有效像素掩膜的样本。各项指标来自同一权重，因而能够反映模型在分类与分割之间实际取得的联合表现。总体结果见表5-1。''')
labels={'binary_iou':'IoU','binary_dice':'Dice','image_macro_f1':'图像宏平均F1','group_macro_f1':'分组宏平均F1','joint_score':'综合指标J'}
rows=[]
for key,label in labels.items():
 r=next(r for r in D['summary'] if r['metric']==key);rows.append([label,fmt(r['plain_mean'],r['plain_sample_sd']),fmt(r['lgla_mean'],r['lgla_sample_sd']),f"{r['relative_gain_percent']:.2f}%"])
b.append(tab('5-1','开发验证集三种子主比较',['指标','Plain','LGLA','相对变化'],rows,'n=3，均值±样本标准差；相对变化按三种子均值计算。'))
b+=ps('''LGLA的图像宏平均F1由0.54578提高至0.55945，病灶分组宏平均F1由0.61492提高至0.63993，综合指标J由0.68524提高至0.70224。按三种子均值计算，相对增幅分别为2.50%、4.07%和2.48%。其中，分组宏平均F1的绝对增加量为0.02501；换成百分数表述，即由61.492%提高至63.993%，增加约2.501个百分点。这一结果说明，在既定数据与训练条件下，加入区域引导和融合约束后，分类描述符的整体辨别能力有所改善。
图5-1进一步呈现每个种子的变化。三个配对运行的图像宏平均F1、分组宏平均F1及J均沿改善方向变化，平均收益并非完全由一个较高的运行造成。分组宏平均F1的配对增量分别为0.01150、0.02564和0.03789，表明收益幅度仍受训练随机性的影响。分割IoU的均值达到0.77066，三个种子中有两个高于对应基线。不同指标的样本标准差变化并不一致，因此这些结果支持分类与综合指标的跨种子改善，不能据此判断所有任务的训练波动都减小。''')
b.append(fig('5-1','三个随机种子的配对结果','paired_results','依据各运行既定权重对应的原始CSV行绘制；各面板使用局部纵轴范围。'))
b+=ps('''图5-2选取相同种子43的两次运行，绘制原始逐轮记录中的验证IoU、分组宏平均F1和J。曲线保留实际训练终止点，空心圆标出各自既定权重的位置。Plain在可见第60轮获得本次报告的综合结果，LGLA对应第99轮。曲线中的起伏说明，联合任务的较优状态并不必然出现在同一训练阶段，也说明单独观察分割峰值不足以评价分类分支的改进效果。''')
b.append(fig('5-2','种子43的原始验证指标训练轨迹','training_curves','横轴为从1开始的训练轮次；空心圆为既定权重，曲线未经平滑。'))
b+=ps('''上述比较中，P3/P4分类分支与部分标签训练流程均已包含在Plain中，因此差值对应LGLA融合结构及其均衡约束的整体变化，不能再分配为多尺度分支或缺失标签屏蔽的独立收益。三个随机种子用于观察重复运行的一致性，本研究未据此给出统计显著性结论。''');sec('基础模型与改进模型总体比较',b)
b=ps('''为比较不同特征选择方式，表5-2汇总了相同YOLO26m任务适配基础上的ECA、ECA+Spatial7、CBAM与LGLA。前期注意力实验使用相同开发清单、输入尺度、分类损失权重、主要优化参数和增强设置；指标同样取各自既定综合指标权重。Plain、ECA、ECA+Spatial7和LGLA均具有三个种子，CBAM只有种子43，因此将其明确列为单次运行，不附加跨种子标准差。''')
rows=[]
for family in ['Plain','ECA','ECA+Spatial7','CBAM','LGLA']:
 runs=[r for r in D['development']+D['historical_attention'] if r['method']==family];rows.append([family,str(len(runs))]+[fmt(statistics.mean([r['metrics'][m] for r in runs]),statistics.stdev([r['metrics'][m] for r in runs]) if len(runs)>1 else None,True) for m in ['binary_iou','image_macro_f1','group_macro_f1','joint_score']])
b.append(tab('5-2','不同注意力方案的开发验证结果',['方法','n','IoU','图像F1','分组F1','J'],rows,'多种子项为均值±样本标准差；CBAM仅seed43。F1均为宏平均。'))
b+=ps('''ECA的三种子平均IoU为0.76684，分组宏平均F1为0.60451，J为0.68080。加入Spatial7后，分组宏平均F1为0.60677，J为0.68096。LGLA的对应结果为0.63993和0.70224。在这些已有方案中，LGLA获得了更高的分类与综合指标均值。通道或空间重标定能够改变局部特征响应，但随后采用全局平均汇聚时，仍需把整个特征图压缩成单个描述符。LGLA则显式保留全局与病灶区域两路描述，使局部信息的汇聚过程能够利用分割输出。这是两类方法的结构差别，也是解释结果的一种合理方向；现有比较并未直接测量各通道的语义含义，不能把该解释视为已隔离验证的因果机制。
前期方案也体现了评价目标的差异。ECA+Spatial7相对ECA在三个种子的MEL误判为NV比例上均较低，其综合指标均值与ECA接近，未呈现与该错误类型变化同等明确的整体收益。因此，注意力模块的选择应同时考察分类、分割及关注的错误类型，不能仅凭某一指标判断整体优劣。CBAM在种子43上的J为0.68534，该结果用于补充结构比较，不与三种子均值形成同等强度的稳定性结论。''');sec('不同注意力方案的比较',b)
b=ps('''在完整结构的基础上，设置无均衡约束、常量区域引导和固定等权融合三种对照，考察区域信息与融合策略的作用。相关实验均采用种子43，结果见表5-3。这些实验用于检验具体结构改变后的表现，并非与主比较相同规模的三种子消融。''')
rows=[]
for run,label in [('22','Plain'),('lg1','无均衡约束'),('ab1','常量区域引导'),('ab2','固定等权融合'),('lgx','完整LGLA')]:
 r=next(r for r in D['development'] if r['run']==run);rows.append([label,'Run22' if run=='22' else run.upper()]+[fmt(r['metrics'][m]) for m in ['binary_iou','image_macro_f1','group_macro_f1','joint_score']])
b.append(tab('5-3','种子43的消融与结构对照',['方法','运行','IoU','图像F1','分组F1','J'],rows,'每行取同一既定权重；AB1、AB2的同时变化因素见正文。'))
b+=ps('''无均衡约束的初始融合保留病灶引导局部描述与可学习门控，但未在训练目标中加入批次级门控均衡项。其图像宏平均F1为0.53761，分组宏平均F1为0.61057，J为0.68105；完整LGLA的相应结果为0.55681、0.65375和0.71346。该对照表明，仅加入两路描述和可学习融合，并不能保证在本次训练中获得有效收益。加入均衡约束后，门控学习所对应的结果更好，这与减轻训练过程长期偏向单一路径的设计动机相符。不过，单次训练结果不能替代对优化全过程的因果分析，均衡项的效果仍需结合区域引导和分类监督共同理解。
常量区域引导对照用全图恒定空间权重替代病灶概率图，并保留其余参数结构与均衡项。此时局部池化退化为与全局平均池化等价的空间汇聚，因而不仅移除了病灶位置，还同时削弱了两路描述之间的差异。其分组宏平均F1为0.61436，J为0.68653。与完整结构相比，这一对照支持非均匀区域权重对当前融合方案有实际作用，但不能把全部差值归因于“精确定位”这一个因素，因为双路互补性也随之改变。
固定等权融合对照采用0.5与0.5合并两路描述，并关闭门控均衡项，分类与分割任务保持原有设置。其分组宏平均F1为0.59544，J为0.67648。完整结构在本种子上取得更好的结果，支持保留自适应融合及其约束的组合设计。该对照同时移除了可学习门控的有效计算和相应训练约束，不能宣称它单独隔离了“是否自适应”之外的全部因素。
综合三项对照可以看到，完整LGLA在既定种子下取得了更好的分类与联合结果。表中各对照仍能完成分割，分类指标差异并不是由取消分割任务造成。这些结果为完整结构的选择提供依据，但尚不能对各组成部分在其他种子或其他数据域上的独立增益作出保证。''');sec('消融实验与结构对照',b)
b=ps('''为观察病灶概率引导在具体图像上的表现，图5-3展示验证集中的四个已有样本。图中依次为原始图像、真实掩膜、预测病灶概率及概率叠加结果，使用完整LGLA种子43的既定第99轮权重。概率图是分割分支输出经Sigmoid及空间插值得到的连续数值，不是分类梯度解释结果；图示可以说明局部池化所用区域依据的形态，不能据此认定分类决策完全由图中高响应区域决定。''')
b.append(fig('5-3','验证样本的真实掩膜与病灶概率图','qualitative','已有预测：seed43，epoch_0098（可见第99轮）。概率范围[0,1]，未二值阈值化；第三、四例缺分类标签。'))
b+=ps('''样本ISIC_0000173具有MEL分类标签。预测概率覆盖了主要深色病灶，并在边缘形成连续过渡，所存分类预测为MEL，全局与局部门控权重约为0.53和0.47。样本ISIC_0000187具有NV标签，预测概率覆盖较大范围的病灶，所存分类预测为NV，两路权重约为0.71和0.29。两例的融合比例不同，直观说明完整模型并未强制每张图像采用等权融合。它们只是选定样本的表现，不能替代完整验证集的宏平均指标。
样本ISIC_0000038和ISIC_0000049有分割掩膜而缺少可用疾病分类标签，只能比较区域形态，不能根据其预测类别判断分类正确与否。前一例中，概率图呈现病灶主体，细小边缘与真实掩膜仍有差别；后一例中，真实掩膜覆盖范围较宽，预测概率更集中于中央色素区域，外围较浅部分的覆盖并不一致。此类边界差别会影响局部描述采集到的信息量，也说明利用预测区域进行汇聚时，分割不确定性可能继续影响分类特征。
从结构上看，停止梯度能够限制分类损失直接改变病灶概率的路径，但不能消除区域预测本身的误差。全局分支保留周围信息，为局部描述不足时提供补充；它的存在也意味着模型可能利用背景或成像条件相关特征。现有样本未包含独立归因实验，因此对于背景依赖的判断仍应谨慎。进一步分析分类机制时，应分别检查区域预测、融合权重与分类贡献，避免把一个可视化对象当作三者的共同证明。''');sec('预测结果可视化与误差分析',b)
b=ps('''LGLA在分类汇聚位置引入额外门控计算，需要同时考虑性能收益与资源开销。表5-4给出了已有同机测量记录中的结果。Plain共有14,507,338个参数，完整LGLA共有14,540,748个参数，新增33,410个，占基线参数量约0.23%。这一增量相对于共享特征网络较小，与利用已有分割输出并采用紧凑尺度投影的实现方式一致。''')
b.append(tab('5-4','参数规模与同机推理开销',['方法','参数量','直接前向/ms','预测入口/ms'],[['Plain','14,507,338','5.4703','6.2254'],['LGLA','14,540,748','5.6642','6.5456']],'种子43代表权重，单机FP32测量。两种计时边界不同，不与网页耗时混用。'))
b+=ps('''耗时测量采用NVIDIA GeForce RTX 5080。直接前向输入为1×3×640×640，精度为FP32，批量大小为1，采用CUDA事件计时，预热20次后测量100次；边界仅包含模型前向，不包含图像预处理、后处理和主机传输。Plain与LGLA的平均直接前向耗时分别为5.4703 ms和5.6642 ms。LGLA增加0.1939 ms，在该环境下相对增加约3.54%，说明分类收益伴随着一定计算成本。
另一测量边界是对同一600×450内存图像调用预测入口，配置imgsz=640，预热5次并测量30次，平均耗时分别为6.2254 ms和6.5456 ms。该结果包含预处理、后处理与同步，入口的实际填充形状未记录，不能按直接前向的方形张量条件解释。这里将两种边界分开列示。这些结果来自种子43代表权重的单机工程测量，不是三种子训练结果的统计，也不直接代表其他硬件上的部署性能。''');sec('参数规模与推理开销',b)
b=ps('''在开发验证比较之外，使用既定权重对固定内部测试集合及外部增量分割集合进行补充评价。内部集合包含634张有分割标签的图像，其中只有56张具有可用分类标签，且真实类别仅含MEL 16张和NV 40张；外部增量集合包含89张图像，用于分割评价。两轮评价沿用相同清单，因此这些结果属于既有固定测试数据上的补充比较，不是新的独立盲测。所有正式结果保持分割阈值0.5、分类温度1.0和八类最大概率决策。表5-5列出两个集合上的分割结果。''')
rows=[]
for scope,label in [('internal_segmentation','内部测试（634）'),('external_segmentation','外部增量（89）')]:
 for test,method in [('test1','Plain'),('test2','LGLA')]:
  vals=[next(z for z in D['test_recomputed'] if z['test']==test and z['section']==scope and z['metric']==m) for m in ['binary_iou','binary_dice']];rows.append([label,method]+[fmt(v['mean'],v['sample_sd']) for v in vals])
b.append(tab('5-5','固定测试集合的补充分割结果',['集合（图像数）','方法','IoU','Dice'],rows,'三种子均值±样本标准差；每次运行先累计有效像素，再计算指标。'))
b+=ps('''LGLA在内部测试集合上的IoU达到0.78558，Dice为0.87990，与基础模型结果接近。在89张外部增量图像上，LGLA的IoU为0.59449±0.00671，Dice为0.74567±0.00528。这些数值描述了既定模型在相应样本范围上的表现；外部集合来自公开数据增量，不能等同于独立医院的临床验证。
受限分类结果见表5-6。虽然真实标签只覆盖MEL和NV，分类器仍在八个类别中作出决策，因此预测到其他类别同样计为错误，而不是重新训练一个二分类器。''')
rows=[]
for test,method in [('test1','Plain'),('test2','LGLA')]:
 vals=[next(z for z in D['test_recomputed'] if z['test']==test and z['section']=='restricted_classification' and z['metric']==m) for m in ['accuracy','mel_recall','nv_recall']];rows.append([method]+[fmt(v['mean'],v['sample_sd']) for v in vals])
b.append(tab('5-6','内部56张标签子集的受限分类结果',['方法','准确率','MEL召回率','NV召回率'],rows,'三种子均值±样本标准差；真实MEL 16张、NV 40张；保留八类argmax决策。'))
b+=ps('''LGLA的受限准确率为0.79167±0.02062，MEL召回率为0.43750±0.06250，NV召回率为0.93333±0.01443。两个类别的召回率存在差别，总体准确率不能完整反映困难类别的识别情况。尤其在类别数量不平衡的子集中，较多类别的正确预测会对总体值产生较大影响，需要同时查看各类召回率。
图5-4根据已保存的种子43分类预测逐条计数。完整LGLA将16个MEL样本中的6个判为MEL、10个判为NV；40个NV样本中37个判为NV、3个判为MEL。这一记录指出MEL与NV之间的混淆仍是值得关注的误差类型。该子集缺少其余六类的真实标签，不能用于宣称完整八类的独立测试宏平均F1，也不据此报告八类综合指标J。''')
b.append(fig('5-4','种子43在受限分类子集上的实际混淆计数','restricted_confusion','源于固定内部测试56条保存预测；温度1.0，八类argmax，epoch_0098。'));sec('固定测试集与外部增量集的补充结果',b)
b=ps('''为展示改进模型的图像分析结果，已有本地界面调用训练后的LGLA权重，将病灶区域和类别概率放在同一图像上呈现。图5-5给出一个单图演示实例：输入样本为ISIC_0024674，界面显示病灶轮廓及八类概率，并将NV列为最高概率类别。该界面用于观察模型输出，不参与模型选择或正式指标计算。
演示采用LG3种子44的best.pt，其推理权重对应既定第97轮权重经部署精度转换后的参数。截图中的分割阈值为0.40，与正式实验固定使用的0.5不同；演示阈值仅影响界面显示的分割区域，不能用该截图替代正式分割结果。类别概率来自单次模型前向，不等同于经过独立校准的诊断可信度。界面中的毫米标尺为示意元素，没有物理尺寸标定依据。这个实例表明已训练模型能够通过本地界面展示分割区域与类别输出，为检查个体样本提供直观入口；算法性能仍以本章的成组实验为依据。''')
b.append(fig('5-5','LGLA的本地单图预测展示','deployment','已有截图；LG3 seed44，ISIC_0024674，演示分割阈值0.40。'));sec('模型部署与可视化演示',b)
sec('本章小结',ps('''本章从总体比较、注意力方案比较、结构对照、可视化与计算开销等方面验证了LGLA。开发验证集上的三种子结果表明，图像宏平均F1、病灶分组宏平均F1和综合指标J的均值均有所提高，三个配对运行方向一致。种子43的结构对照支持完整区域引导与受约束融合组合的选择，真实概率图展示了该结构使用的区域信息及其边界误差。新增参数量约0.23%，同机直接前向平均耗时为5.6642 ms。固定测试与外部增量结果补充了不同样本范围上的性能描述，后续研究仍需关注困难类别、区域不确定性及独立数据验证。'''))
(O/'chapter_5_results.json').write_text(json.dumps({'chapters':[ch]},ensure_ascii=False,indent=2),encoding='utf-8')
ch6={'number':6,'title':'总结及展望','sections':[{'title':'研究工作总结','blocks':ps('''本文围绕皮肤病图像中病灶局部信息与全局上下文的利用问题，在YOLO26m语义分割与八类分类的任务适配基础上，设计并实现了病灶引导全局—局部注意力融合模块。研究重点位于共享特征的汇聚阶段：利用主分割输出构建软区域权重，对P3与P4特征分别提取全局平均描述和病灶引导局部描述，在同尺度共享投影的条件下形成两路256维表示，再通过可学习门控进行样本自适应融合。
为使区域引导与联合优化的关系清晰可控，方法在病灶概率路径上设置停止梯度，同时保留分类与分割对共享特征网络的更新。针对两路融合的批次整体倾向，在具有分类标签的样本上引入门logit均值差的平方约束。该约束不要求每张图像等权融合。训练流程继续采用标签可用性屏蔽机制，分别在有效像素集合和有效分类样本集合上计算损失，从而利用分割与分类标注覆盖不同的异构图像。
在固定开发验证集上，Plain与完整LGLA的三种子比较显示，图像宏平均F1由0.54578提高到0.55945，病灶分组宏平均F1由0.61492提高到0.63993，综合指标J由0.68524提高到0.70224，相对增幅分别约为2.50%、4.07%和2.48%。这三项指标在三个配对种子中均有改善。单种子结构对照进一步表明，完整区域引导与受约束融合组合取得了较好的联合结果。与基线相比，模型新增33,410个参数，约占0.23%；同机平均直接前向耗时为5.6642 ms。
本文给出了算法定义、张量尺寸、有效标签处理和固定评价规则，并通过真实预测案例观察区域引导的具体形态。已有局部部署演示为查看单样本输出提供了入口。整体结果支持在既定任务与开发数据条件下采用LGLA改进分类特征汇聚，但这一结论仍限定于当前模型规模、标签覆盖与评价集合，不能据此推导所有类别或所有数据域均得到改善。''')},{'title':'后续研究方向','blocks':ps('''后续可首先研究区域引导的不确定性。当前局部描述直接依赖预测概率，边界模糊或浅色病灶可能使有效区域与真实掩膜存在差异。可在保持全局补充路径的基础上，分析概率质量、边界不确定性与融合权重的关系，进一步讨论局部描述应在何种条件下得到加强或回退。这类研究需要独立的归因或受控实验，不能仅依据分割概率图判断分类关注范围。
数据方面，需要扩大分类与分割双监督的覆盖，并补充可靠的病灶和患者关联信息。对于类别间相似性较强、样本数量较少的类别，应结合逐类别错误和成像来源分析表征差异。新的验证数据应覆盖全部目标类别，明确采集来源与患者分组边界；若研究跨来源适配，还应采用与开发过程分离的评价方案，检验方法在独立条件下的适用性。
实验方面，可在预先确定的协议下增加重复运行，并分别检验门控约束、引导质量和融合策略的影响。当前单种子结构对照同时改变部分相关因素，后续可设置更细致的控制实验以区分各环节作用。在模型效率方面，还可研究投影维度、计算精度与部署算子的关系，使用相同硬件和计时边界评估精度与开销的取舍。上述方向均应以新增、独立的验证证据为基础，避免把开发集上的改进直接外推到实际诊疗场景。''')} ]}
abstract_cn='针对皮肤病图像分类中病灶局部特征与全局上下文难以兼顾、分割与分类标签覆盖不一致的问题，本文在YOLO26m语义分割与八类图像分类的任务适配基础上，设计病灶引导全局—局部注意力融合模块（LGLA）。该模块将停止梯度后的分割概率作为软区域权重，对P3、P4多尺度特征进行局部加权汇聚，并与全局平均描述通过共享投影和自适应门控融合；训练时在有分类标签的样本上加入批次级门控均衡约束，按标签可用性计算联合损失。实验采用既定开发数据划分，对Plain与完整LGLA分别进行三个随机种子比较，并结合单种子结构对照、真实预测可视化和固定测试集合补充评价分析方法表现。结果表明，LGLA在开发验证集上的图像宏平均F1、病灶分组宏平均F1和综合指标J分别由0.54578、0.61492和0.68524提高至0.55945、0.63993和0.70224，相对增幅约为2.50%、4.07%和2.48%，且三项指标在三个配对种子中改善方向一致。模型新增参数33,410个，约占基线的0.23%，同机平均直接前向耗时为5.6642 ms。研究表明，利用分割概率组织局部与全局特征融合，可在既定任务和数据条件下改善分类及联合评价表现，为部分任务标签条件下的皮肤图像分析提供一种可复现的改进方法。'
abstract_en='To balance lesion-specific features and global context in skin lesion classification under incomplete task annotations, this study developed a lesion-guided global–local attention module (LGLA) on a YOLO26m model adapted for binary semantic segmentation and eight-class image classification. The module used detached segmentation probabilities as soft spatial weights to pool P3 and P4 features. Local descriptors were combined with global average descriptors through shared projections and an adaptive gate. A batch-level gate-balancing penalty was applied to samples with classification labels, while joint losses were computed according to annotation availability. Plain and full LGLA models were compared using three random seeds on a fixed development split. Single-seed structural controls, saved prediction visualizations, and supplementary evaluation on fixed test sets were also analyzed. On the development validation set, image macro-F1, lesion-group macro-F1, and the joint metric J increased from 0.54578, 0.61492, and 0.68524 to 0.55945, 0.63993, and 0.70224, respectively. The relative gains were approximately 2.50%, 4.07%, and 2.48%, with consistent directions across all three paired seeds. LGLA added 33,410 parameters, approximately 0.23% of the baseline, and had a mean direct-forward latency of 5.6642 ms on the same device. The results support segmentation-guided fusion of local and global features as a reproducible way to improve classification and joint performance under the specified task and data conditions.'
(O/'closing_and_abstract.json').write_text(json.dumps({'chapters':[ch6],'abstract_cn':abstract_cn,'abstract_en':abstract_en,'acknowledgements':'感谢指导教师郑莉萍老师在毕业论文研究和写作过程中给予的指导。围绕研究问题、方法表达与实验分析的学习，使我逐步认识到，算法研究不仅需要得到结果，也需要说明结果如何获得、能够支持怎样的结论。感谢学院老师在专业学习中给予的教导，感谢家人与同学在学习和论文完成过程中给予的理解与支持。今后的学习中，我将继续重视问题分析、实验规范和独立思考。'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved chapter 5, chapter 6 and matched abstracts.')
