# 文献核验与第一二章写作说明

核验日期：2026年11月9日。范围：原始论文、正式出版页、作者主页或仓库、官方数据集与软件文档；未读取旧论文正文。

## 交付文件

- `references.json`：20条文献，按当前第1—2章首次出现次序排列；全篇合并后仍应重新检查首次出现顺序。
- `chapters_1_2.json`：两章独立草稿，标题编号单列，正文含 `[REF:key]` 引用占位；第1章约2777字符，第2章约3318字符（含公式解释，不含公式LaTeX与标题）。
- 第2章提供六条评价公式，需合成时转为可编辑OMML。图 `[FIG:research_route]` 需替换为技术路线图的真实交叉引用。

## 核验边界

已核查下表所列书目信息与正文所使用的核心论点，不等同于逐篇全文复现或研究方法的系统综述。部分CVF链接直接open返回403，但同一官方域的检索回传提供了论文页/BibTeX与PDF原文段落，核验采用这些官方回传。AdamW的OpenReview页面触发浏览器验证，采用作者arXiv v3（明确标注ICLR2019）及作者代码仓库交叉确认。没有使用二手转载作为最终技术论据。

所有正文为独立概括，没有复制论文段落或挪用他文实验值。文献中的U-Net、DeepLabv3+、SE仅解释相关研究，不列为本项目已实施实验；Kendall不确定性加权和Li跨任务一致性也未写为本文采用算法。

## 文献与论点

|顺序|键|使用论点|核验依据与说明|
|---|---|---|---|
|1|esteva|深度卷积网络用于皮肤病变图像分类；细粒度外观差异构成识别困难。|[原始来源](https://www.nature.com/articles/nature21056)；期刊原始页面核对题名、作者、年份、卷页、DOI与摘要；不引用其医生水平结果作为本项目结论。|
|2|ham|皮肤镜公开数据具有多来源、不同采集方式及同病灶多图特点。|[原始来源](https://www.nature.com/articles/sdata2018161)；Scientific Data原文核对完整元数据及数据说明；不将其10015张规模冒充项目冻结规模。|
|3|isic2018|病灶分割与疾病分类构成皮肤图像分析的相关但不同任务；基准成绩与泛化能力应区分。|[原始来源](https://arxiv.org/abs/1902.03368)；作者arXiv原始发布核对题名、作者、提交年份及摘要；保留预印本类型，不虚构会议页码。|
|4|isic2019|ISIC2019训练数据覆盖八个类别，官方挑战另有未知类设定；本文限于项目八类任务。|[原始来源](https://challenge.isic-archive.com/landing/2019/)；官方任务页核对训练八类、测试额外outlier类别及官方评价口径；本文不沿用挑战排名指标。|
|5|unet|编码收缩路径获取上下文，扩张路径恢复定位信息。|[原始来源](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)；Springer正式章节页核对作者、会议、年份、页码及DOI；只作研究背景，不宣称项目开展U-Net对比。|
|6|deeplab|DeepLabv3+结合多尺度上下文与解码恢复细节。|[原始来源](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html)；CVF正式页面与作者arXiv核对作者、题名、ECCV2018页码及方法摘要；未将其写入实测比较。|
|7|ima|IMA++提供多标注者皮肤镜病灶掩膜、标注信息、官方多标注子集划分与共识掩膜。|[原始来源](https://www.sfu.ca/~kabhishe/)；作者SFU主页BibTeX确认正式题名Descriptor前缀、期刊、卷页、DOI；arXiv确认相同作者与journal reference。Zenodo v1.1及作者仓库确认数据构成、STAPLE、许可。|
|8|se|全局汇聚后学习通道权重以校准特征响应。|[原始来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)；CVF论文页核对CVPR版三位作者、题名、页码；没有误用arXiv后续TPAMI版五位作者。|
|9|eca|不降维的局部跨通道交互及一维卷积构成轻量通道注意力。|[原始来源](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)；CVF页及PDF核对题名、六位作者、CVPR2020页码、避免降维与局部跨通道交互。|
|10|cbam|顺序通道和空间注意力通过乘法重标定中间特征。|[原始来源](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)；CVF页与论文PDF核对完整四位作者、题名、年份、页码和模块作用。|
|11|mbdcnn|以粗分割掩膜引导分类，再向分割转移定位信息，是分割和分类协作的先行研究。|[原始来源](https://arxiv.org/abs/1903.03313)；作者arXiv与官方代码核对结构与作者，PubMed期刊记录核对卷期页和DOI；不拿其指标与本项目直接比较。|
|12|caruana|相关任务的共享表征可形成归纳偏置；需要讨论任务关系。|[原始来源](https://doi.org/10.1023/A:1007379606734)；出版商页面核对作者、题名、卷页、DOI及摘要；不将潜在收益当作项目独立消融结果。|
|13|kendall|多任务损失的量级与相对权重影响联合训练；其不确定性加权属于相关工作。|[原始来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)；CVF原始页面核对作者、页码与论文目标；明确本文未采用其不确定性加权。|
|14|partial|每图仅部分任务标签可用的多任务部分监督问题；只在可用标签上计算损失是基本方案。|[原始来源](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Learning_Multiple_Dense_Prediction_Tasks_From_Partially_Annotated_Data_CVPR_2022_paper.html)；CVF页面/PDF核对三位作者与任务定义。页码采用CVF公开版本18879-18889；爱丁堡元数据另记IEEE版18857-18867，辅助记录保留差异，不混配。|
|15|staple|STAPLE在多个分割标注之间估计潜在分割及标注表现，可用于共识掩膜。|[原始来源](https://pmc.ncbi.nlm.nih.gov/articles/PMC1283110/)；PMC作者存档原文核对方法、作者、正式出版卷期页和DOI；姓名按完整首名书写、不猜中间名。|
|16|yolo|YOLO将目标检测组织为单网络直接预测；本文语义分割分类任务不同于原始检测输出。|[原始来源](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)；CVF论文页核对完整作者、题名、年份、页码；不挪用原论文FPS及检测性能。|
|17|fpn|横向连接与自顶向下路径支持不同分辨率的语义特征表达；用于说明多尺度思想。|[原始来源](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)；CVF页面/PDF核对完整题名、作者和公开版页码2117-2125；不把项目结构称为未修改的原FPN。|
|18|semantic|官方区分语义分割和实例分割，YOLO26具有-sem模型；项目历史版本须由本地证据确定。|[原始来源](https://docs.ultralytics.com/tasks/semantic/)；2026-11-09实际访问官方文档；只支持任务概念与模型命名，不用当前默认值替代历史配置。|
|19|dropout|训练时随机屏蔽单元的正则化方法；本文分类头Dropout与门控均衡作用不同。|[原始来源](https://www.jmlr.org/papers/v15/srivastava14a.html)；JMLR原始页面/PDF核对五位作者、卷号/文章号和页码；不宣称独立证明正则化改善。|
|20|adamw|AdamW将权重衰减与自适应梯度更新解耦；为本项目实际优化器的文献依据。|[原始来源](https://arxiv.org/abs/1711.05101)；作者arXiv v3注明ICLR2019，作者代码仓库再次确认；OpenReview遇浏览器验证，未声称已读其页面，不虚构页码。|

## 数据与版本的区别

- IMA++文献介绍的是公开资源整体；项目有效11409对、冻结划分等实施事实由方法代理回到本地清单核验，不能用论文总规模替代。
- IMA++正式题名为“Descriptor: ISIC Archive Multi-Annotator Dermoscopic Skin Lesion Segmentation Dataset (IMA++)”。作者SFU主页完整BibTeX与arXiv journal reference均确认IEEE Data Descriptions 2026年第3卷367—378页，DOI 10.1109/IEEEDATA.2026.3689801。正式题名采用作者主页而非arXiv旧标题。
- IMA++ [作者仓库](https://github.com/sfu-mial/IMAplusplus)代码为Apache-2.0；[Zenodo v1.1数据发布](https://zenodo.org/records/14201693)为CC BY-NC-ND 4.0。两者许可不同；本次不转载官方数据示意图，也不把代码许可推及数据。
- 官方[YOLO语义分割文档](https://docs.ultralytics.com/tasks/semantic/)仅用于任务定义和模型命名。训练记录中8.4.101、预训练权重8.4.50及P3/P4尺寸由method_audit直接核验，并非当前网页推断。YOLO26论文 https://arxiv.org/abs/2606.03748 也已访问，但为遵守学校10—20篇要求，不再纳入与官方文档重复的独立文献。
- Li等部分标注论文存在CVF公开版本页码18879—18889与爱丁堡机构库IEEE元数据页码18857—18867差异；本文采用CVF官方页面及PDF页码，不将不同出版页码混合。
- SENet采用CVPR2018三位作者版本；未使用arXiv后续TPAMI五位作者名单替换会议版。

## 本地方法信息核验协作记录

- method_audit已直接核验：LGX冻结epoch_0098/trainer.pt的Ultralytics版本8.4.101，git branch/commit均None；预训练models/yolo26m-sem.pt版本8.4.50。
- 开发评价：`.venv/Lib/site-packages/ultralytics/utils/metrics.py:1706—1755`累加有效像素2×2混淆矩阵，二值miou为前景IoU；开发Dice=2IoU/(1+IoU)。
- 分类与分组：`tools/imaplusplus_partial_multitask_v1.py:184—195,340—379,449—474`；group_id内Softmax概率平均后argmax，缺失group_id回退isic_id；固定8类F1均值，零分母0；J每运行独立计算。
- results_audit已直接核验：`tools/test2/test.py:405—483`内部使用相同PartialLabelMultiTaskValidator，外部分割继承SemanticSegmentationValidator；442—445行从metrics.miou取得binary_iou并换算Dice。`tools/test1/test.py`约439—442行相同。故第2章没有写成逐图IoU平均。

## 合成前检查

- 当前引用键数：20；参考文献数：20；不存在未定义引用键。
- 英文作者使用全拼的首名和姓，不臆造无法确认的中间名；超过三名时使用et al.，这是作者列表省略而非姓名缩写。
- 首次引用顺序应由全篇程序生成；Word正文引用必须上标并链接真实参考文献书签。
- 六条公式具有LaTeX主体和符号说明；公式可见性、换行及编号需由最终Word渲染验证，本文件不声称已完成排版检查。
- 章节草稿中的研究结论未预填尚未复算的增益；摘要与最终结论由主代理采用结果核验数据统一。

