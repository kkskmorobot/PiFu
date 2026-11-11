from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json, hashlib
ROOT=Path(r'E:\App\DaiMa\JetBrains\Project\PiFuV2\LunWen')
OUT=ROOT/'astra-6-ultra/Data/academic_rewrite'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
paths={'reference':ROOT/'astra-6-ultra/Data/参考.docx','template':ROOT/'Data/计算机学院毕业论文（设计）论文模板-应用开发型（人工智能、智能科学与技术专业适用）-2026版（7-6）.docx','personal':ROOT/'论文-6-极高/毕业论文正文_修订版.docx'}
report={}
for label,path in paths.items():
 with ZipFile(path) as z:
  doc=E.fromstring(z.read('word/document.xml')); body=doc.find('w:body',NS)
  texts=[]
  for i,p in enumerate(body.xpath('.//w:p',namespaces=NS)):
   t=''.join(p.xpath('.//w:t/text()',namespaces=NS))
   if label=='personal' and i>=22: break
   texts.append(f'{i:04d} {t}')
  (OUT/f'{label}_read.txt').write_text('\n'.join(texts),encoding='utf-8')
  if label=='personal': continue
  report[label]={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parts':[{'path':n,'size':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()], 'sections':[E.tostring(e,encoding='unicode') for e in doc.xpath('//w:sectPr',namespaces=NS)],'fields':doc.xpath('//w:instrText/text()',namespaces=NS),'equations':len(doc.xpath('//m:oMath',namespaces=NS)),'comments':[],'headers_footers':{},'direct_format_examples':[]}
  if 'word/comments.xml' in z.namelist():
   c=E.fromstring(z.read('word/comments.xml'))
   report[label]['comments']=[{'id':e.get('{'+NS['w']+'}id'),'text':''.join(e.xpath('.//w:t/text()',namespaces=NS))} for e in c]
  for n in z.namelist():
   if n.startswith('word/header') or n.startswith('word/footer'): report[label]['headers_footers'][n]=z.read(n).decode()
  report[label]['styles']=z.read('word/styles.xml').decode()
  for p in body.xpath('.//w:p',namespaces=NS):
   t=''.join(p.xpath('.//w:t/text()',namespaces=NS))
   if t: report[label]['direct_format_examples'].append({'text':t,'xml':E.tostring(p,encoding='unicode')})
(OUT/'document_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print([(k,len(v['direct_format_examples']),len(v['comments']),len(v['sections']),v['equations']) for k,v in report.items()])
print('TEMPLATE COMMENTS')
for c in report['template']['comments']: print(c)
print('PERSONAL');print((OUT/'personal_read.txt').read_text(encoding='utf-8'))

