from pathlib import Path
import json,csv,shutil,math
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon,Circle,PolyLine
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor,Color,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
P=Path(r'E:\App\DaiMa\JetBrains\Project\PiFuV2');O=P/'LunWen/astra-6-ultra/Data/academic_rewrite';F=O/'figures';F.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('SimHei',r'C:\Windows\Fonts\simhei.ttf'))
D=json.loads((O/'results_data.json').read_text(encoding='utf-8'));MD=json.loads((O/'method_data.json').read_text(encoding='utf-8'))
C={'ink':'#182633','blue':'#356783','teal':'#258379','orange':'#B07337','pale':'#EEF3F6','green':'#EAF5F0','muted':'#55616D','gray':'#D4DDE3'}
def color(x):return HexColor(C.get(x,x))
def text(d,x,y,s,size=22,anchor='middle',fill='ink'):
 for k,line in enumerate(s.split('\n')):d.add(String(x,d.height-y-k*(size*1.25),line,fontName='SimHei',fontSize=size,textAnchor=anchor,fillColor=color(fill)))
def box(d,x,y,w,h,s,fill='pale',size=22,stroke='blue'):
 d.add(Rect(x,d.height-y-h,w,h,rx=5,ry=5,fillColor=color(fill),strokeColor=color(stroke),strokeWidth=1.5));lines=s.split('\n'); text(d,x+w/2,y+h/2-(len(lines)-1)*size*.625+size*.33,s,size)
def arrow(d,x1,y1,x2,y2,dash=False,col='muted'):
 l=Line(x1,d.height-y1,x2,d.height-y2,strokeColor=color(col),strokeWidth=1.8)
 if dash:l.strokeDashArray=[6,4]
 d.add(l);a=math.atan2(y2-y1,x2-x1);p=[x2,d.height-y2,x2-10*math.cos(a-.4),d.height-(y2-10*math.sin(a-.4)),x2-10*math.cos(a+.4),d.height-(y2-10*math.sin(a+.4))];d.add(Polygon(p,fillColor=color(col),strokeColor=color(col)))
def save(d,name):
 d.contents.insert(0,Rect(0,0,d.width,d.height,fillColor=white,strokeColor=None));renderSVG.drawToFile(d,str(F/(name+'.svg')));renderPDF.drawToFile(d,str(F/(name+'.pdf')))
# 图1.1 技术路线
x=Drawing(900,370)
for a,s in [(20,'数据与监督\nIMA++分割 / ISIC分类'),(330,'任务适配基线\n共享特征 + P3/P4分类'),(640,'LGLA改进\n区域池化 + 门控均衡')]:box(x,a,25,240,90,s,size=23)
arrow(x,260,70,330,70);arrow(x,570,70,640,70)
for a,s in [(20,'固定数据与训练条件\n部分任务标签损失屏蔽'),(330,'三种子主比较\n单种子结构对照'),(640,'结果与开销分析\n固定集合补充评价')]:box(x,a,180,240,90,s,'green',23,'teal')
arrow(x,140,115,140,180);arrow(x,450,115,450,180);arrow(x,760,115,760,180);arrow(x,260,225,330,225);arrow(x,570,225,640,225)
text(x,450,335,'研究主线：病灶区域如何参与特征汇聚，以及融合策略能否改善分类与联合指标',22)
save(x,'research_route')
# 图3.1 总体结构
x=Drawing(900,430)
box(x,20,20,200,80,'RGB输入\nB×3×640×640',size=21);box(x,280,20,280,80,'共享YOLO26m\n骨干与多尺度特征融合',size=23);arrow(x,220,60,280,60)
box(x,625,20,255,80,'P3：256×80×80\nP4：512×40×40',size=22);arrow(x,560,60,625,60)
x.add(Line(752,x.height-100,752,x.height-135,strokeColor=color('muted'),strokeWidth=1.8));x.add(Line(140,x.height-135,752,x.height-135,strokeColor=color('muted'),strokeWidth=1.8))
for a in [140,450,752]:arrow(x,a,135,a,170)
box(x,20,170,240,100,'语义分支（保留）\nP3→主输出S\nP4→辅助输出（训练）',size=20)
box(x,330,170,240,100,'Plain分类路径\nGAP(P3/P4)→拼接768\nMLP 768→256→8',size=20)
box(x,630,170,250,100,'LGLA分类路径\n区域引导融合→256\nMLP 256→256→8','green',20,'teal')
x.add(Line(140,x.height-270,140,x.height-320,strokeColor=color('teal'),strokeWidth=1.8,strokeDashArray=[6,4]));x.add(Line(140,x.height-320,752,x.height-320,strokeColor=color('teal'),strokeWidth=1.8,strokeDashArray=[6,4]));arrow(x,752,320,752,270,True,'teal')
text(x,450,355,'S经过停止梯度与Sigmoid，仅引导LGLA局部汇聚',22,fill='teal')
text(x,450,400,'Plain与LGLA为对比模型；共享网络、语义头与标签监督流程保持一致',21)
save(x,'architecture')# 图3.2 细节：顶端概率分支，中央每尺度池化，底部门控
x=Drawing(900,610)
box(x,30,20,190,65,'主分割logit S',size=22);box(x,290,20,230,65,'stop-gradient\nSigmoid',size=22);box(x,590,20,275,65,'双线性插值→A3/A4\nalign_corners=False',size=21);arrow(x,220,52,290,52);arrow(x,520,52,590,52)
box(x,30,150,180,110,'P3 / P4特征\n分别处理\n256 / 512通道',size=21)
box(x,280,120,230,75,'全局平均池化\n全局描述 g',size=22);box(x,280,235,230,100,'概率加权池化\n分母保护 / 小质量回退\n局部描述 l',size=20)
arrow(x,210,185,280,157);arrow(x,210,235,280,285);x.add(PolyLine([727,x.height-85,727,x.height-105,550,x.height-105,550,x.height-265],strokeColor=color('teal'),strokeWidth=1.8,strokeDashArray=[6,4]));arrow(x,550,265,510,265,True,'teal')
box(x,590,130,275,100,'每尺度共享投影→128\nP3、P4投影相互独立\n两尺度各自拼接',size=21);arrow(x,510,158,590,163);arrow(x,510,285,590,207)
box(x,610,285,230,70,'G：256维\nL：256维',size=22);arrow(x,727,230,727,285)
box(x,40,405,270,105,'门网络 [G;L]：512\nLinear128→SiLU→Linear2\n得到logit q','green',20,'teal');arrow(x,610,320,175,405)
box(x,380,405,210,105,'Softmax\nα + β = 1\n每样本自适应','green',22,'teal');arrow(x,310,457,380,457)
box(x,650,405,210,105,'融合 f=αG+βL\n256维\n送入分类器','green',22,'teal');arrow(x,590,457,650,457);arrow(x,755,355,755,405)
text(x,450,564,'均衡项作用于q的批次平均差；仅有分类标签样本参与',22)
text(x,450,595,'概率图路径停止梯度；分类与分割仍共享前面的特征网络',22,fill='muted');save(x,'lgla_detail')
# 图3.3 部分标签联合损失
x=Drawing(900,390)
box(x,20,25,235,95,'一个混合批次\n分割标签 / 分类标签\n可同时存在或缺失',size=22)
box(x,335,20,235,75,'has_seg筛选有效像素V\n缺失掩膜255忽略',size=20);box(x,335,170,235,75,'has_cls筛选样本C\n缺失类别−1屏蔽',size=20)
arrow(x,255,60,335,58);arrow(x,255,85,335,205)
box(x,640,15,235,95,'BCE + Dice\n+ 0.4×辅助BCE\n→ Lseg',size=22);box(x,640,160,235,100,'C上分类CE\nC上一次批次均衡R\n→ Lcls + 0.01R',size=21)
arrow(x,570,58,640,58);arrow(x,570,207,640,207)
box(x,220,300,620,65,'总损失：B×[Lseg + 0.5×(Lcls + 0.01R)]','green',25,'teal');x.add(PolyLine([757,x.height-110,893,x.height-110,893,x.height-285,820,x.height-285],strokeColor=color('muted'),strokeWidth=1.8));arrow(x,820,285,820,300);arrow(x,757,260,757,300)
text(x,136,180,'双监督样本\n进入两条路径',21);save(x,'partial_loss')
# 图4.1 标签覆盖与类比例
x=Drawing(900,505)
text(x,450,24,'监督覆盖比例（各集合内归一化）',23)
for y,split,label in [(65,'train','训练'),(125,'val','验证')]:
 a=MD['development'][split];counts=[a['seg']-a['both'],a['both'],a['cls']-a['both']];start=150
 text(x,85,y+22,label,22)
 for val,col in zip(counts,['blue','orange','teal']):
  width=710*val/a['total'];x.add(Rect(start,x.height-y-35,width,35,fillColor=color(col),strokeColor=white));start+=width
 text(x,505,y+58,'仅分割 '+str(counts[0])+'  |  双监督 '+str(counts[1])+'  |  仅分类 '+str(counts[2]),19)
text(x,450,211,'条内比例按各集合图像总数归一化；标签数值为图像数量',18,fill='muted')
text(x,450,251,'有分类标签图像的类别占比（%）',23)
classes=['akiec','bcc','bkl','df','mel','nv','vasc','scc'];ys=450;h=165
for t in [0,10,20,30,40]:
 yy=ys-t/45*h;x.add(Line(70,x.height-yy,875,x.height-yy,strokeColor=color('gray')));text(x,55,yy+5,str(t),17,'end')
for i,k in enumerate(classes):
 xx=100+i*100
 for off,split,col in [(-15,'train','blue'),(15,'val','teal')]:
  a=MD['development'][split];v=a['class_counts'][k]/a['cls']*100;bh=v/45*h;x.add(Rect(xx+off-12,x.height-ys,24,bh,fillColor=color(col),strokeColor=None))
 text(x,xx,478,k.upper(),18)
text(x,650,270,'蓝：训练   绿：验证',19);save(x,'data_distribution')
# 图4.2 实验流程
x=Drawing(900,350)
box(x,20,20,250,70,'相同冻结开发清单\n相同任务与主要训练设置',size=22)
box(x,340,20,250,70,'Plain / 完整LGLA\n各seed 42、43、44',size=22);arrow(x,270,55,340,55)
box(x,340,145,250,80,'无均衡 / 常量引导\n固定等权\n均为seed43',size=21);arrow(x,145,90,340,185)
box(x,655,20,220,100,'既定J权重\n同轮多指标\n均值与样本SD','green',22,'teal');arrow(x,590,55,655,55)
box(x,655,175,220,70,'单次结构对照\n说明控制因素','green',22,'teal');arrow(x,590,185,655,207)
text(x,450,303,'补充评价：相同内部634张 / 外部89张；固定条件、既有测试集合',22)
text(x,450,335,'主比较与结构对照分开解释，不将六次新增运行混为重复实验',21,fill='muted');save(x,'experiment_flow')
# 图5.1 三seed配对，局部轴显式说明
x=Drawing(900,650);metrics=[('image_macro_f1','图像宏平均F1',.535,.565),('group_macro_f1','分组宏平均F1',.60,.66),('joint_score','综合指标J',.675,.72),('binary_iou','验证IoU',.76,.785)]
colors=['blue','teal','orange']
for idx,(m,title,lo,hi) in enumerate(metrics):
 ox=30+(idx%2)*450;oy=55+(idx//2)*290;left=ox+90;right=ox+365;top=oy+25;bottom=oy+210
 text(x,ox+225,oy,title,23)
 for k in range(4):
  val=lo+(hi-lo)*k/3;y=bottom-(val-lo)/(hi-lo)*(bottom-top);x.add(Line(left,x.height-y,right,x.height-y,strokeColor=color('gray')));text(x,left-12,y+5,f'{val:.3f}',18,'end')
 for seed,col in zip([42,43,44],colors):
  a=next(r for r in D['development'] if r['method']=='Plain' and r['seed']==seed)['metrics'][m];b=next(r for r in D['development'] if r['method']=='LGLA' and r['seed']==seed)['metrics'][m]
  yy=lambda v:bottom-(v-lo)/(hi-lo)*(bottom-top)
  x.add(Line(left+30,x.height-yy(a),right-30,x.height-yy(b),strokeColor=color(col),strokeWidth=2.5));x.add(Circle(left+30,x.height-yy(a),5,fillColor=color(col),strokeColor=None));x.add(Circle(right-30,x.height-yy(b),5,fillColor=color(col),strokeColor=None))
 text(x,left+30,bottom+30,'Plain',20);text(x,right-30,bottom+30,'LGLA',20)
for i,(s,col) in enumerate(zip([42,43,44],colors)):text(x,190+i*250,623,'● seed '+str(s),21,fill=col)
text(x,450,648,'各面板采用已标注的局部纵轴范围，连线表示同一种子配对',19,fill='muted');save(x,'paired_results')
# 图5.2 原始训练轨迹
x=Drawing(900,575)
for idx,(m,title) in enumerate([('binary_iou','验证IoU'),('group_macro_f1','分组宏平均F1'),('joint_score','综合指标J')]):
 top=40+idx*170;bottom=top+125;left=105;right=865;text(x,485,top-13,title,20)
 for v in [0,.25,.5,.75,1]:
  y=bottom-v*125;x.add(Line(left,x.height-y,right,x.height-y,strokeColor=color('gray')));text(x,left-12,y+5,f'{v:.2f}',16,'end')
 for ep in ([0,25,50,75,100] if idx==2 else []):text(x,left+ep/100*(right-left),bottom+24,str(ep),17)
 for run,col in [('22','blue'),('lgx','teal')]:
  r=next(r for r in D['development'] if r['run']==run);pts=[]
  for a in r['curve']:pts.extend([left+a['epoch']/100*(right-left),x.height-(bottom-a[m]*125)])
  x.add(PolyLine(pts,strokeColor=color(col),strokeWidth=1.8));val=r['metrics'][m];cx=left+r['epoch']/100*(right-left);cy=x.height-(bottom-val*125);x.add(Circle(cx,cy,5,fillColor=white,strokeColor=color(col),strokeWidth=2.2))
text(x,270,562,'蓝：Plain seed43（93轮）',20,fill='blue');text(x,650,562,'绿：LGLA seed43（100轮）',20,fill='teal');save(x,'training_curves')
# 真实受限分类矩阵2x8，从保存CSV重数；不反推
x=Drawing(900,280);s=next(z for z in D['test_results']['test2']['data']['per_seed'] if z['seed']==43);cm=s['saved_csv_confusion'];classes=['akiec','bcc','bkl','df','mel','nv','vasc','scc']
text(x,470,30,'固定内部测试：真实类别2类，模型预测空间8类（seed43）',22)
for i,y in enumerate(['mel','nv']):
 text(x,63,122+i*70,y.upper(),23)
 for k,pred in enumerate(classes):
  v=cm[y][pred];xx=110+k*95;yy=70+i*70;shade=1-v/40*.78;x.add(Rect(xx,x.height-yy-65,90,65,fillColor=Color(.25+.65*shade,.45+.5*shade,.55+.4*shade),strokeColor=white));text(x,xx+45,yy+42,str(v),24)
for k,pred in enumerate(classes):text(x,155+k*95,235,pred.upper(),20)
text(x,470,272,'横轴：预测类别；纵轴：真实类别；单元格为保存预测的实际样本数',19,fill='muted');save(x,'restricted_confusion')
for src,name in [(P/'tools/sum2/figure_attention_heatmaps_seed43.png','qualitative.png'),(P/'LunWen/Data/单张预测示意图1.png','deployment.png')]:shutil.copyfile(src,F/name)
(O/'figures_data.json').write_text(json.dumps({'method_data':'method_data.json','results_data':'results_data.json','generated':['research_route','architecture','lgla_detail','partial_loss','data_distribution','experiment_flow','paired_results','training_curves','restricted_confusion'],'unchanged_images':{'qualitative':str(P/'tools/sum2/figure_attention_heatmaps_seed43.png'),'deployment':str(P/'LunWen/Data/单张预测示意图1.png')},'note':'所有新图为ReportLab矢量绘图；曲线和矩阵使用已保存记录，未运行模型。'},ensure_ascii=False,indent=2),encoding='utf-8')
print('9 vector figures and 2 unchanged existing images prepared.')

