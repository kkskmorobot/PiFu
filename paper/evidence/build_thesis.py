from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import json,re,hashlib,statistics
from docx import Document
from docx.shared import Pt,Cm,RGBColor,Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK,WD_LINE_SPACING,WD_TAB_ALIGNMENT,WD_TAB_LEADER
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from math_omml import to_omml
O=Path(__file__).parent;ROOT=O.parents[2];TEMPLATE=ROOT/'Data/计算机学院毕业论文（设计）论文模板-应用开发型（人工智能、智能科学与技术专业适用）-2026版（7-6）.docx';FINAL=O.parents[1]/'毕业论文正文_算法研究重写版.docx'
CH=[]
for f in ['chapters_1_2.json','chapters_3_4.json','chapter_5_results.json','closing_and_abstract.json']:CH+=json.loads((O/f).read_text(encoding='utf-8'))['chapters']
CLOSE=json.loads((O/'closing_and_abstract.json').read_text(encoding='utf-8'));REFS=json.loads((O/'references.json').read_text(encoding='utf-8'));RD=json.loads((O/'results_data.json').read_text(encoding='utf-8'))
TITLE='基于注意力机制改进YOLO的皮肤病识别算法研究'
# Normalize section numbering and map figure placeholders.
figmap={'research_route':('1-1','research_route'),'3-1':('3-1','architecture'),'3-2':('3-2','lgla_detail'),'3-3':('3-3','partial_loss'),'4-1':('4-1','data_distribution'),'4-2':('4-2','experiment_flow')}
for ci,ch in enumerate(CH,1):
 ch['number']=ci
 for si,s in enumerate(ch['sections'],1):
  s.setdefault('number',f'{ci}.{si}');s.setdefault('level',2)
  for b in s['blocks']:
   if b['type']=='figure_placeholder':b.update(type='figure',file='figures/'+figmap[b['id']][1]+'.png',source='依据本项目实际实现与冻结数据绘制。')
   if b['type']=='p':
    b['text']=b['text'].replace('qᵢ,ɢ,qᵢ,ʟ','qᵢ,G、qᵢ,L')
    if '[FIG:research_route]' in b['text']:b['text']=b['text'].replace('[FIG:research_route]','[FIG:1-1]')
  if ci==1 and s['number']=='1.3':s['blocks'].append({'type':'figure','id':'1-1','title':'算法研究技术路线','file':'figures/research_route.png','source':'围绕数据、模型改进与实验验证绘制。'})
# First occurrence reference order.
used=[]
for ch in CH:
 for s in ch['sections']:
  for b in s['blocks']:
   for k in re.findall(r'\[REF:([^]]+)\]',b.get('text','')+b.get('explanation','')):
    if k not in used:used.append(k)
ref_by_key={r['key']:r for r in REFS};assert set(used)==set(ref_by_key),(set(ref_by_key)-set(used));refnums={k:i+1 for i,k in enumerate(used)}
D=Document(TEMPLATE);body=D._element.body
base_section=deepcopy(D.sections[0]._sectPr)
for e in list(body)[8:]:body.remove(e)
body.append(base_section)
# All source content after cover header is an authorized replacement slot.
for rid,rel in list(D.part.rels.items()):
 if any(k in rel.reltype.lower() for k in ['comment','people','customxml']):D.part.drop_rel(rid)
settings=D.settings.element
for tag in ['w:evenAndOddHeaders','w:trackRevisions']:
 for e in settings.findall(qn(tag)):settings.remove(e)
u=OxmlElement('w:updateFields');u.set(qn('w:val'),'true');settings.append(u)
# Font and paragraph rules, explicit values override inconsistent template styles.
def fonts(obj,east='宋体',size=12,bold=False):
 f=obj.font;f.name='Times New Roman';f.size=Pt(size);f.bold=bold;f.color.rgb=RGBColor(0,0,0)
 el=obj.element if hasattr(obj,'element') else obj._element
 rp=el.get_or_add_rPr();rf=rp.find(qn('w:rFonts'))
 if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
 for a,v in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('eastAsia',east),('cs','Times New Roman')]:rf.set(qn('w:'+a),v)
 for k in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
  rf.attrib.pop(qn('w:'+k),None)
def clean_style(s):
 for tag in ['w:numPr','w:tabs','w:ind','w:spacing','w:pBdr']:
  p=s.element.get_or_add_pPr()
  for e in p.findall(qn(tag)):p.remove(e)
 for tag in ['w:u','w:color']:
  r=s.element.get_or_add_rPr()
  for e in r.findall(qn(tag)):r.remove(e)
def style(name,east,size,bold=False,align=WD_ALIGN_PARAGRAPH.JUSTIFY,line=20,before=0,after=0,indent=0):
 s=D.styles[name] if name in D.styles else D.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH);clean_style(s);fonts(s,east,size,bold);p=s.paragraph_format;p.alignment=align;p.line_spacing=Pt(line);p.space_before=Pt(before);p.space_after=Pt(after);p.first_line_indent=Pt(indent);p.left_indent=Pt(0);p.right_indent=Pt(0);p.widow_control=True;return s
style('Normal','宋体',12,indent=24)
for lev,font,size,bold in [(1,'黑体',15,False),(2,'宋体',14,True),(3,'楷体',12,False)]:
 s=style(f'Heading {lev}',font,size,bold,WD_ALIGN_PARAGRAPH.CENTER if lev<3 else WD_ALIGN_PARAGRAPH.LEFT,20,20 if lev<3 else 0,20 if lev==1 else 0);s.paragraph_format.keep_with_next=True;s.paragraph_format.keep_together=True
style('Title','黑体',16,False,WD_ALIGN_PARAGRAPH.CENTER,24,20,20)
# Title must not enter TOC.
for e in D.styles['Title'].element.get_or_add_pPr().findall(qn('w:outlineLvl')):D.styles['Title'].element.pPr.remove(e)
style('Caption','黑体',10.5,False,WD_ALIGN_PARAGRAPH.CENTER,16,4,4)
style('Figure','宋体',12,False,WD_ALIGN_PARAGRAPH.CENTER,20,4,0);D.styles['Figure'].paragraph_format.line_spacing=1
style('Note','宋体',10.5,False,WD_ALIGN_PARAGRAPH.LEFT,16,0,5)
style('Equation','宋体',12,False,WD_ALIGN_PARAGRAPH.LEFT,40,0,0)
style('Bibliography','宋体',12,False,WD_ALIGN_PARAGRAPH.LEFT,20,0,0)
style('Back Heading','黑体',15,False,WD_ALIGN_PARAGRAPH.CENTER,20,20,20);D.styles['Back Heading'].paragraph_format.keep_with_next=True
ol=OxmlElement('w:outlineLvl');ol.set(qn('w:val'),'0');D.styles['Back Heading'].element.get_or_add_pPr().append(ol)
for lev in [1,2,3]:
 name=next((s.name for s in D.styles if s.name.lower()==f'toc {lev}'),f'TOC {lev}');s=style(name,'黑体' if lev==1 else '宋体',14 if lev==1 else 12,False,WD_ALIGN_PARAGRAPH.LEFT,20);s.paragraph_format.left_indent=Pt((lev-1)*12)
style('Header','宋体',10.5,False,WD_ALIGN_PARAGRAPH.LEFT,14)
style('Footer','宋体',10.5,False,WD_ALIGN_PARAGRAPH.CENTER,14)
# Word fields and bookmarks.
bm_id=1000

def field(p,instruction,cache='',sup=False):
 if sup:instruction+=' \\* CHARFORMAT'
 def run():
  r=OxmlElement('w:r');rp=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'21' if sup else str(round((p.style.font.size.pt if p.style.font.size else 12)*2)));rp.append(sz)
  if sup:v=OxmlElement('w:vertAlign');v.set(qn('w:val'),'superscript');rp.append(v)
  r.append(rp);p._p.append(r);return r
 r=run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'begin');r.append(e)
 r=run();e=OxmlElement('w:instrText');e.set(qn('xml:space'),'preserve');e.text=' '+instruction+' ';r.append(e)
 r=run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'separate');r.append(e)
 r=run();e=OxmlElement('w:t');e.text=cache;r.append(e)
 r=run();e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'end');r.append(e)
def bm_start(p,name):
 global bm_id
 bm_id+=1;e=OxmlElement('w:bookmarkStart');e.set(qn('w:id'),str(bm_id));e.set(qn('w:name'),name);p._p.append(e);return bm_id
def bm_end(p,i):
 e=OxmlElement('w:bookmarkEnd');e.set(qn('w:id'),str(i));p._p.append(e)
def literal(p,s):
 # Typeset source inline dimensional superscripts instead of visible TeX braces.
 pos=0
 for m in re.finditer(r'R\^\{([^}]+)\}|(?<![A-Za-z_])(?:TP_k|FP_k|FN_k|n_g|p_i|p_\{i,k\}|F_\{1,group\}|x_s)(?![A-Za-z_])',s):
  p.add_run(s[pos:m.start()])
  if m.group(1):
   p.add_run('ℝ');r=p.add_run(m.group(1));r.font.superscript=True
  else:p._p.append(to_omml(m.group(0)))
  pos=m.end()
 p.add_run(s[pos:])
def rich(p,s):
 s=re.sub(r'图\s*(\d+)[.-](\d+)',lambda m:f'[FIG:{m[1]}-{m[2]}]',s)
 s=re.sub(r'表\s*([A-Z]|\d+)[.-](\d+)',lambda m:f'[TABLE:{m[1]}-{m[2]}]',s)
 s=re.sub(r'式（(\d+-\d+)）',lambda m:'式[EQ:'+m[1]+']',s)
 pos=0
 for m in re.finditer(r'\[(REF|FIG|TABLE|EQ):([^]]+)\]',s):
  literal(p,s[pos:m.start()]);kind,k=m.groups()
  if kind=='REF':field(p,'REF Ref_'+k+' \\h',f'[{refnums[k]}]',True)
  elif kind=='EQ':field(p,'REF Eq_'+k.replace('-','_')+' \\h','（'+k+'）')
  else:field(p,'REF '+('Fig_' if kind=='FIG' else 'Tab_')+k.replace('-','_')+' \\h',('图' if kind=='FIG' else '表')+k.replace('-','.'))
  pos=m.end()
 literal(p,s[pos:])
def para(text,sty=None):
 p=D.add_paragraph(style=sty or 'Normal');rich(p,text);p.paragraph_format.widow_control=True
 if '都减小' in text:p.paragraph_format.keep_together=True
 return p

def section_settings(s,kind):
 s.page_width=Twips(11906);s.page_height=Twips(16838);s.top_margin=s.bottom_margin=s.left_margin=Twips(1418);s.right_margin=Twips(1134);s.header_distance=s.footer_distance=Twips(851);s.different_first_page_header_footer=False
 for e in list(s._sectPr):
  if E.QName(e).localname in ['headerReference','footerReference'] and e.get(qn('w:type'))!='default':s._sectPr.remove(e)
 for e in s._sectPr.findall(qn('w:docGrid')):s._sectPr.remove(e)
 s.header.is_linked_to_previous=False;s.footer.is_linked_to_previous=False
 for h in [s.header,s.footer]:
  for e in list(h._element):h._element.remove(e)
 hp=s.header.add_paragraph(style='Header');fp=s.footer.add_paragraph(style='Footer')
 pg=s._sectPr.find(qn('w:pgNumType'))
 if pg is not None:s._sectPr.remove(pg)
 if kind!='cover':
  hp.paragraph_format.tab_stops.add_tab_stop(Cm(16.5),WD_TAB_ALIGNMENT.RIGHT)
  hp.add_run('四川大学锦江学院本科毕业论文');hp.add_run('\t');hp.add_run(TITLE)
  for r in hp.runs:fonts(r,'宋体',10.5)
  pb=OxmlElement('w:pBdr');bt=OxmlElement('w:bottom');bt.set(qn('w:val'),'single');bt.set(qn('w:sz'),'4');bt.set(qn('w:space'),'3');bt.set(qn('w:color'),'000000');pb.append(bt);hp._p.get_or_add_pPr().append(pb)
  field(fp,'PAGE','I' if kind=='front' else '1');pg=OxmlElement('w:pgNumType');pg.set(qn('w:fmt'),'upperRoman' if kind=='front' else 'decimal');pg.set(qn('w:start'),'1');s._sectPr.append(pg)
section_settings(D.sections[0],'cover')
# Original logo and cover title retained; cover form rows rebuilt to hold the longer title.
for pp in D.paragraphs[:8]:
 for e in pp._p.xpath('.//w:commentRangeStart | .//w:commentRangeEnd | .//w:commentReference'):e.getparent().remove(e)
pcover=D.paragraphs[6];pcover.style='Title'
for r in pcover.runs:fonts(r,'华文宋体',36,False)
pcover.paragraph_format.space_before=Pt(0);pcover.paragraph_format.space_after=Pt(0);pcover.paragraph_format.line_spacing=Pt(44)
t=D.add_table(rows=6,cols=2);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False;t.columns[0].width=Cm(3.4);t.columns[1].width=Cm(9.4)
cover=[('题    目','基于注意力机制改进YOLO的\n皮肤病识别算法研究'),('学    院','计算机学院'),('专    业','人工智能    年级  2023级'),('学生姓名','陈孔旭'),('学    号','2305500413'),('指导教师','郑莉萍')]
for row,(a,b) in zip(t.rows,cover):
 for i,tx in enumerate([a,b]):
  cell=row.cells[i];cell.width=Cm(3.4 if i==0 else 9.4);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;p=cell.paragraphs[0];p.text=tx;p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.left_indent=Pt(0);p.paragraph_format.right_indent=Pt(0);p.paragraph_format.line_spacing=Pt(22);p.paragraph_format.space_before=Pt(9);p.paragraph_format.space_after=Pt(9)
  for r in p.runs:fonts(r,'宋体',14)
  borders=OxmlElement('w:tcBorders')
  for side in ['top','left','bottom','right']:
   e=OxmlElement('w:'+side);e.set(qn('w:val'),'single' if i==1 and side=='bottom' else 'nil');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'000000');borders.append(e)
  cell._tc.get_or_add_tcPr().append(borders)
# Abstracts and TOC.
front=D.add_section(WD_SECTION_START.NEW_PAGE);section_settings(front,'front')
p=para(TITLE,'Title');p.paragraph_format.line_spacing=Pt(24)
for tx in ['人工智能专业','学生：陈孔旭    指导教师：郑莉萍']:
 p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(20)
 r=p.add_run(tx);fonts(r,'楷体',14)
p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(42);p.paragraph_format.right_indent=Pt(21);r=p.add_run('【摘 要】');fonts(r,'楷体',10.5,True);r=p.add_run(CLOSE['abstract_cn']);fonts(r,'楷体',10.5)
p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(42);p.paragraph_format.right_indent=Pt(21)
for tx,bold in [('【关键词】',True),('皮肤病图像；YOLO26；病灶引导注意力；部分标签多任务学习；语义分割',False)]:r=p.add_run(tx);fonts(r,'楷体',10.5,bold)
p=para('Research on Skin Disease Recognition Algorithms\nBased on Attention-Enhanced YOLO','Title');p.paragraph_format.page_break_before=True
for r in p.runs:fonts(r,'Times New Roman',16,True)
p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(24);p.paragraph_format.right_indent=Pt(24);p.add_run('Abstract: ').bold=True;p.add_run(CLOSE['abstract_en'])
p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(24);p.paragraph_format.right_indent=Pt(24);p.add_run('Keywords: ').bold=True;p.add_run('skin lesion images; YOLO26; lesion-guided attention; partially labeled multi-task learning; semantic segmentation')
p=para('目  录','Title');p.paragraph_format.page_break_before=True
p=D.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);field(p,'TOC \\o "1-3" \\h \\z \\u','目录')
main=D.add_section(WD_SECTION_START.NEW_PAGE);section_settings(main,'main')
# Captions and content components.
figures=[];tables=[];equations=[]
def caption(kind,id,title):
 ch,n=id.split('-');p=D.add_paragraph(style='Caption');p.paragraph_format.keep_with_next=kind=='表';i=bm_start(p,('Fig_' if kind=='图' else 'Tab_')+id.replace('-','_'));p.add_run(kind+ch+'.');field(p,'SEQ '+('Fig' if kind=='图' else 'Tab')+ch+' \\* ARABIC',n);bm_end(p,i);p.add_run('  '+title);return p

def table(b):
 caption('表',b['id'],b['title']);t=D.add_table(rows=1,cols=len(b['headers']));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 n=len(b['headers']);widths=([5.4,11.1] if n==2 else [5.5,5.5,5.5] if n==3 else [4.1,4.1,4.1,4.2] if n==4 else [4.1,2.4,2,3.8,4.2] if n==5 else [3.6,1.1,2.95,2.95,2.95,2.95])
 if b['id']=='5-1':widths=[3.4,4.8,4.8,3.5]
 if b['id']=='5-2':widths=[3.0,.7,3.2,3.2,3.2,3.2]
 if b['id']=='5-3':widths=[3.5,1.5,2.875,2.875,2.875,2.875]
 if b['id']=='4-7':widths=[4.3,2.5,2.1,3.8,3.8]
 for j,w in enumerate(widths):t.columns[j].width=Cm(w)
 for i,rowdata in enumerate([b['headers']]+b['rows']):
  row=t.rows[0] if i==0 else t.add_row();trpr=row._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');trpr.append(ns)
  if i==0:rep=OxmlElement('w:tblHeader');trpr.append(rep)
  for j,tx in enumerate(rowdata):
   c=row.cells[j];c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;p=c.paragraphs[0];p.text=str(tx);p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=Pt(15);p.paragraph_format.space_before=Pt(2 if b['id']=='4-4' else 3);p.paragraph_format.space_after=Pt(2 if b['id']=='4-4' else 3);p.paragraph_format.keep_with_next=bool(b.get('source')) and i==len(b['rows']);p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or j>0 else WD_ALIGN_PARAGRAPH.LEFT
   for r in p.runs:fonts(r,'黑体' if i==0 else '宋体',10.5)
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    e=OxmlElement('w:'+side);visible=(i==0 and side in ['top','bottom']) or (i==len(b['rows']) and side=='bottom');e.set(qn('w:val'),'single' if visible else 'nil');e.set(qn('w:sz'),'8' if i==0 and side=='top' or i==len(b['rows']) and side=='bottom' else '4');e.set(qn('w:color'),'000000');borders.append(e)
   c._tc.get_or_add_tcPr().append(borders)
   mar=OxmlElement('w:tcMar')
   for side in ['left','right']:e=OxmlElement('w:'+side);e.set(qn('w:w'),'65');e.set(qn('w:type'),'dxa');mar.append(e)
   c._tc.get_or_add_tcPr().append(mar)
 if b.get('source'):para('注：'+b['source'],'Note')
 tables.append({'id':b['id'],'title':b['title'],'rows':b['rows']})

def figure(b):
 p=D.add_paragraph(style='Figure');p.paragraph_format.keep_with_next=True;p.paragraph_format.keep_together=True;r=p.add_run();pic=r.add_picture(str(O/b['file']),width=Cm(16.3));pic._inline.docPr.set('descr',b['title']);caption('图',b['id'],b['title']).paragraph_format.keep_with_next=bool(b.get('source'))
 if b.get('source'):para('注：'+b['source'],'Note')
 figures.append({'id':b['id'],'title':b['title'],'file':b['file']})

def equation(b):
 formula=b['latex'];split=(b['label'] in ['2-3','2-4','2-6','3-2','3-5','3-6','3-7','3-8','3-11','3-12']);parts=re.split(r',\\q(?:quad|uad)\s*',formula) if split else [formula]
 for index,s in enumerate(parts):
  p=D.add_paragraph(style='Equation');p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST;p.paragraph_format.space_after=Pt(6);p.paragraph_format.keep_with_next=True;p.paragraph_format.tab_stops.add_tab_stop(Cm(8.25),WD_TAB_ALIGNMENT.CENTER);p.paragraph_format.tab_stops.add_tab_stop(Cm(16.5),WD_TAB_ALIGNMENT.RIGHT);p.add_run('\t');math=to_omml(s);p._p.append(math)
  if index==len(parts)-1:
   p.add_run('\t');i=bm_start(p,'Eq_'+b['label'].replace('-','_'));ch,n=b['label'].split('-');p.add_run('（'+ch+'-');field(p,'SEQ Eq'+ch+' \\* ARABIC',n);p.add_run('）');bm_end(p,i)
 if b.get('explanation'):para(b['explanation'])
 equations.append({'id':b['label'],'latex':formula,'lines':len(parts)})
for ch in CH:
 p=para(f"{ch['number']} {ch['title']}",'Heading 1');p.paragraph_format.page_break_before=ch['number']!=1
 for s in ch['sections']:
  p=para(s['number']+' '+s['title'],'Heading '+str(s['level']))
  if s['number']=='5.7':p.paragraph_format.page_break_before=False
  for b in s['blocks']:
   if b.get('explanation'):b['explanation']=b['explanation'].replace('qᵢ,ɢ,qᵢ,ʟ','qᵢ,G、qᵢ,L')
   if b['type']=='p':para(b['text'])
   elif b['type']=='table':table(b)
   elif b['type']=='figure':figure(b)
   elif b['type']=='equation':equation(b)
   else:raise ValueError(b['type'])
p=para('参考文献','Back Heading');p.paragraph_format.page_break_before=True
for k in used:
 p=D.add_paragraph(style='Bibliography');p.paragraph_format.left_indent=Pt(25);p.paragraph_format.first_line_indent=Pt(-25);i=bm_start(p,'Ref_'+k);p.add_run(f'[{refnums[k]}]');bm_end(p,i);p.add_run(' '+ref_by_key[k]['formatted']);p.paragraph_format.keep_together=True
p=para('附录1  既定实验权重对应关系','Back Heading');p.paragraph_format.page_break_before=True;p.alignment=WD_ALIGN_PARAGRAPH.LEFT
para('为便于对应实验，表A-1列出主比较与结构对照所使用的既定权重。可见轮次从1开始，权重目录的epoch从0开始；二者相差1。本文所有同一运行的报告指标均取该权重对应结果。','Note')
b={'id':'A-1','title':'主要运行与既定权重','headers':['方法','运行','种子','权重目录'],'rows':[],'source':'所列目录均位于各运行的checkpoint_bundles下。'}
for r in RD['development']:
 method={'Plain':'Plain','LGLA':'完整LGLA','No balance':'无均衡约束','Constant guidance':'常量引导','Fixed equal fusion':'固定等权'}[r['method']];b['rows'].append([method,'Run'+r['run'] if r['run'].isdigit() else r['run'].upper(),r['seed'],r['bundle']])
table(b)
p=para('致  谢','Back Heading');p.paragraph_format.page_break_before=True;para(CLOSE['acknowledgements'])
D.core_properties.title=TITLE;D.core_properties.author='陈孔旭';D.core_properties.subject='本科算法研究论文';D.core_properties.keywords='LGLA, YOLO26, 皮肤病图像, 多任务学习';D.core_properties.comments=''
# Remove source comments anchors globally; no changes to originals.
for e in D._element.xpath('.//w:commentRangeStart | .//w:commentRangeEnd | .//w:commentReference'):e.getparent().remove(e)
used_relids={e.get(qn('r:id')) for e in D._element.xpath('.//w:headerReference | .//w:footerReference | .//a:blip')}
used_relids.update(e.get(qn('r:embed')) for e in D._element.xpath('.//a:blip'))
for rid,rel in list(D.part.rels.items()):
 if rel.reltype.rsplit('/',1)[-1] in ['header','footer','image'] and rid not in used_relids:D.part.drop_rel(rid)
if FINAL.exists() and not (O/'build_manifest.json').exists():raise RuntimeError('Refuse to overwrite a pre-existing unknown final thesis.')
D.save(FINAL)
# Explicitly remove unreferenced comment/people/custom-XML remnants and old template text-bearing headers.
with ZipFile(FINAL) as z:parts={i.filename:z.read(i.filename) for i in z.infolist()}
for name in list(parts):
 if 'comments' in name or name=='word/people.xml' or name.startswith('customXml/'):del parts[name]
for name in ['[Content_Types].xml','word/_rels/document.xml.rels']:
 root=E.fromstring(parts[name])
 for el in list(root):
  v=' '.join(el.attrib.values())
  if any(s in v for s in ['comments','people.xml','customXml']):root.remove(el)
 parts[name]=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(FINAL,'w',ZIP_DEFLATED) as z:
 for n,v in parts.items():z.writestr(n,v)
manifest={'final':str(FINAL),'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),'figures':figures,'tables':tables,'equations':equations,'references':[{'number':refnums[k],**ref_by_key[k]} for k in used],'chapters':[{ 'number':ch['number'],'title':ch['title'],'prose_chars':sum(len(b.get('text',''))+len(b.get('explanation','')) for s in ch['sections'] for b in s['blocks'])} for ch in CH]}
(O/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(O/'full_manuscript.json').write_text(json.dumps(CH,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(FINAL),'figures':len(figures),'numbered_tables':len(tables),'equations':len(equations),'references':len(used),'chapter_chars':manifest['chapters']},ensure_ascii=False))
