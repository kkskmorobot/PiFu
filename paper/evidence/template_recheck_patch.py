from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re,json,shutil,sys
sys.stdout.reconfigure(encoding='utf8')
O=Path(__file__).parent;Q=O/'template_recheck_20261109';Q.mkdir(exist_ok=True);FINAL=O.parents[1]/'毕业论文正文_算法研究重写版.docx';BASE=Q/'复核前论文.docx'
if not BASE.exists():shutil.copy2(FINAL,BASE);shutil.copy2(FINAL.with_suffix('.pdf'),Q/'复核前论文.pdf')
D=Document(BASE);changes=[]
def font(run,size=12,east='宋体',bold=None):
 run.font.name='Times New Roman';run.font.size=Pt(size)
 if bold is not None:run.bold=bold
 rp=run._r.get_or_add_rPr();rf=rp.find(qn('w:rFonts'))
 if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
 for k,v in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('eastAsia',east),('cs','Times New Roman')]:rf.set(qn('w:'+k),v)
def xmlrun(text,sup=False):
 r=OxmlElement('w:r');rp=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'21' if sup else '24');rp.append(sz)
 if sup:e=OxmlElement('w:vertAlign');e.set(qn('w:val'),'superscript');rp.append(e)
 r.append(rp);t=OxmlElement('w:t');t.text=text;r.append(t);return r
for p in D.paragraphs:
 if p.text=='毕业论文（设计）':
  ind=p._p.get_or_add_pPr().find(qn('w:ind'))
  if ind is not None:p._p.pPr.remove(ind)
  p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.left_indent=Pt(0);p.paragraph_format.right_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  for r in p.runs:font(r,36,'华文宋体',False)
  changes.append('封面主标题清除继承的72磅首行缩进，按模板居中')
 if p.text=='基于注意力机制改进YOLO的皮肤病识别算法研究':
  p.clear();r=p.add_run('基于注意力机制改进YOLO的\n皮肤病识别算法研究');font(r,16,'黑体',False);p.paragraph_format.line_spacing=Pt(24)
  changes.append('中文摘要长标题按4/5版心限制分成两行')
 if p.text.startswith('Abstract: '):
  p.runs[0].text=p.runs[0].text.replace('Abstract: ','【Abstract】 ');p.paragraph_format.first_line_indent=Pt(48);p.paragraph_format.right_indent=Pt(24)
  changes.append('英文摘要首行改为四格48磅，标识对齐模板')
 if p.text.startswith('Keywords: '):
  p.runs[0].text=p.runs[0].text.replace('Keywords: ','【Key words】 ');p.paragraph_format.first_line_indent=Pt(48);p.paragraph_format.right_indent=Pt(24)
  changes.append('英文关键词首行改为四格48磅，标识对齐模板')
 if p.text.startswith('附录1 ') and p.style.name=='Back Heading':
  p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.paragraph_format.space_before=Pt(0)
  for r in p.runs:font(r,14,'黑体',False)
  changes.append('附录标题改为黑体四号14磅、版心左上')
 if p.style.name=='Equation':
  p.paragraph_format.line_spacing=Pt(40);p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.EXACTLY
# Native math remains editable; use compact side limits instead of overgrown stacked sums.
for math in D._element.xpath('.//m:oMath'):
 for pr in math.findall('.//'+qn('m:naryPr')):
  for key,val in [('grow','0'),('limLoc','subSup')]:
   e=pr.find(qn('m:'+key))
   if e is None:e=OxmlElement('m:'+key);pr.append(e)
   e.set(qn('m:val'),val)
 for r in math.findall('.//'+qn('m:r')):
  if ''.join(e.text or '' for e in r.findall(qn('m:t')))=='∈':continue
  mrp=r.find(qn('m:rPr'))
  if mrp is None:mrp=OxmlElement('m:rPr');r.insert(0,mrp)
  sty=mrp.find(qn('m:sty'));plain=sty is not None and sty.get(qn('m:val'))=='p'
  for tag in ['m:nor']:
   e=mrp.find(qn(tag))
   if e is None:e=OxmlElement(tag);mrp.append(e)
   e.set(qn('m:val'),'1')
  rp=r.find(qn('w:rPr'))
  if rp is None:rp=OxmlElement('w:rPr');r.insert(1 if r.find(qn('m:rPr')) is not None else 0,rp)
  txt=''.join(r.itertext());it=rp.find(qn('w:i'))
  if it is None:it=OxmlElement('w:i');rp.append(it)
  it.set(qn('w:val'),'1' if not plain and any(c.isalpha() for c in txt) else '0')
  rf=rp.find(qn('w:rFonts'))
  if rf is None:rf=OxmlElement('w:rFonts');rp.insert(0,rf)
  for k in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+k),'Times New Roman')
  for k in ['sz','szCs']:
   e=rp.find(qn('w:'+k))
   if e is None:e=OxmlElement('w:'+k);rp.append(e)
   e.set(qn('w:val'),'24')
changes.append('公式字母与数字明确设Times New Roman小四，保留原生对象与已有固定40磅；求和上下限紧凑排布')
# Every numbered table fits a page; keep its complete rows and following note together.
for table in D.tables[1:]:
 for i,row in enumerate(table.rows):
  for cell in row.cells:
   for p in cell.paragraphs:
    p.paragraph_format.keep_with_next=True
    for r in p.runs:font(r,10.5,'宋体',i==0)
changes.append('编号表格整体留在同页，消除无续表标识的跨页表；表内列名统一五号宋体加粗，名称仍为黑体')
# Preserve original reference bookmarks; add inner numeric bookmarks for grouped citations.
maxid=max(int(e.get(qn('w:id'))) for e in D._element.xpath('.//w:bookmarkStart'))
num_by_key={}
for p in D.paragraphs:
 starts=[e for e in p._p if e.tag==qn('w:bookmarkStart') and e.get(qn('w:name'),'').startswith('Ref_')]
 if not starts:continue
 old=starts[0];key=old.get(qn('w:name'))[4:];number=re.match(r'\[(\d+)\]',p.text).group(1);num_by_key[key]=int(number)
 r=next(e for e in list(p._p)[list(p._p).index(old)+1:] if e.tag==qn('w:r'));idx=list(p._p).index(r);p._p.remove(r)
 maxid+=1;bs=OxmlElement('w:bookmarkStart');bs.set(qn('w:id'),str(maxid));bs.set(qn('w:name'),'RefNum_'+key);be=OxmlElement('w:bookmarkEnd');be.set(qn('w:id'),str(maxid))
 for off,e in enumerate([xmlrun('['),bs,xmlrun(number),be,xmlrun(']')]):p._p.insert(idx+off,e)
 # Conference paper type follows the school's explicit A / C convention.
 if '[C]//' in p.text:
  new=p.text.split('] ',1)[1];a,b=new.split('[C]//',1)
  b=re.sub(r'\. (?=(?:Cham:|20\d\d))','[C]. ',b,count=1)
  formatted=a+'[A]. '+b
  # All source reference text follows the enclosing Ref_ bookmarkEnd.
  oldend=next(e for e in p._p if e.tag==qn('w:bookmarkEnd') and e.get(qn('w:id'))==old.get(qn('w:id')))
  for e in list(p._p)[list(p._p).index(oldend)+1:]:p._p.remove(e)
  p.add_run(' '+formatted)
changes.append('10条会议论文按模板析出文献[A]、会议文集[C]模式统一著录')
def fieldblocks(p):
 children=list(p._p);out=[];start=None;level=0
 for i,e in enumerate(children):
  for fc in e.iter(qn('w:fldChar')):
   typ=fc.get(qn('w:fldCharType'))
   if typ=='begin':
    if level==0:start=i
    level+=1
   if typ=='end':
    level-=1
    if level==0 and start is not None:
     block=children[start:i+1];instr=''.join(t.text or '' for q in block for t in q.iter(qn('w:instrText')));m=re.search(r'REF Ref_(\w+) ',instr)
     if m:out.append((start,i,m.group(1),block))
 return children,out
for p in D.paragraphs:
 children,blocks=fieldblocks(p)
 for a,b in reversed(list(zip(blocks,blocks[1:]))):
  gap=children[a[1]+1:b[0]]
  if any(''.join(e.itertext()).strip() for e in gap):continue
  ia,ib=num_by_key[a[2]],num_by_key[b[2]]
  replacement=[xmlrun('[',True)]
  for index,f in enumerate([a,b]):
   if index:replacement.append(xmlrun('-' if ib==ia+1 else ',',True))
   block=deepcopy(f[3])
   for e in block:
    for t in e.iter(qn('w:instrText')):t.text=t.text.replace('REF Ref_','REF RefNum_')
    for t in e.iter(qn('w:t')):
     if t.text:t.text=t.text.strip('[]')
   replacement.extend(block)
  replacement.append(xmlrun(']',True))
  for e in children[a[0]:b[1]+1]:p._p.remove(e)
  for offset,e in enumerate(replacement):p._p.insert(a[0]+offset,e)
  changes.append(f'同处文献引用[{ia}][{ib}]合并为上标[{ia}-{ib}]，保留交叉引用')
D.save(FINAL)
(Q/'corrections.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(changes,ensure_ascii=False,indent=2))
