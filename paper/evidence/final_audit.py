from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from PIL import Image,ImageChops
import json,re,hashlib,sys
import pdfplumber,pypdfium2 as pdfium
sys.stdout.reconfigure(encoding='utf8')
O=Path(__file__).parent;ROOT=O.parents[1];DX=ROOT/'毕业论文正文_算法研究重写版.docx';PDF=DX.with_suffix('.pdf');R=O/'final_render_v3';R.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with pdfplumber.open(PDF) as pdf:
 texts=[p.extract_text() or '' for p in pdf.pages]
 (O/'final_pdf_text.txt').write_text('\n\n'.join(f'=== PAGE {i+1} ===\n{t}' for i,t in enumerate(texts)),encoding='utf8')
 pdf_pages=len(pdf.pages)
 headings=[{'page':i+1,'heading':line} for i,t in enumerate(texts) for line in t.splitlines() if re.match(r'^[1-6] [\u4e00-\u9fff]{2,12}$',line)]
 footer=[t.splitlines()[-1] for t in texts]
 assert footer[1:5]==['I','II','III','IV'],footer[1:5]
 assert footer[5:]==list(map(str,range(1,pdf_pages-4))),footer[5:]
 assert all('四川大学锦江学院本科毕业论文' in t.replace(' ','') for t in texts[1:])
 assert not any(b in '\n'.join(texts) for b in ['错误!','错误！','Error!','[REF:','[FIG:','[TABLE:','[EQ:'])
with ZipFile(DX) as z:
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};xml=E.fromstring(z.read('word/document.xml'))
 fields=[e.text for e in xml.findall('.//w:instrText',ns)];bookmarks={e.get('{'+ns['w']+'}name') for e in xml.findall('.//w:bookmarkStart',ns)}
 refs=[re.search(r'REF\s+(\S+)',f).group(1) for f in fields if re.match(r'\s*REF\s',f)];missing=[r for r in refs if r not in bookmarks];assert not missing,missing
 assert sum(bool(re.match(r'\s*SEQ Eq',f)) for f in fields)==18
 assert sum(bool(re.match(r'\s*SEQ Fig',f)) for f in fields)==11
 assert sum(bool(re.match(r'\s*SEQ Tab',f)) for f in fields)==14
 assert sum(bool(re.match(r'\s*TOC ',f)) for f in fields)==1
 assert not any('comments' in n for n in z.namelist())
 bodytxt=''.join(xml.itertext());assert not any(x in bodytxt for x in ['[REF:','[FIG:','[TABLE:','[EQ:'])
 math_count=len(xml.findall('.//m:oMath',ns));assert math_count>=30
 equations=len([x for x in bookmarks if x and x.startswith('Eq_')]);assert equations==18
 data={'pages':pdf_pages,'chapters':headings,'roman_pages':4,'arabic_pages':pdf_pages-5,'fields':len(fields),'cross_references':len(refs),'missing_bookmarks':missing,'native_math_objects':math_count,'numbered_equations':equations,'figures':11,'numbered_tables':14,'references':20,'comments':0}
src=json.loads((O/'document_source_audit.json').read_text(encoding='utf8'));data['sources_unchanged']={k:sha(Path(v['source']))==v['sha256'] for k,v in src.items()};assert all(data['sources_unchanged'].values())
chs=json.loads((O/'full_manuscript.json').read_text(encoding='utf8'));counts=[sum(len(b.get('text',''))+len(b.get('explanation','')) for s in ch['sections'] for b in s['blocks']) for ch in chs];data['chapter_prose_chars']=counts;data['chapters_3_to_5_ratio']=sum(counts[2:5])/sum(counts)
deploy=next(s for s in chs[4]['sections'] if s['number']=='5.7');depl=''.join(b.get('text','') for b in deploy['blocks']);data['deployment_chars']=len(depl);data['deployment_han_chars']=len(re.findall('[\u4e00-\u9fff]',depl));data['deployment_ratio']=len(depl)/sum(counts);assert data['chapters_3_to_5_ratio']>.6 and .003<data['deployment_ratio']<.03 and 300<=data['deployment_han_chars']<=500
pdf=pdfium.PdfDocument(str(PDF));changed=[]
for i in range(len(pdf)):
 pg=pdf[i];bm=pg.render(scale=2);im=bm.to_pil();dst=R/f'page-{i+1}.png';im.save(dst);old=O/f'final_render_v2/page-{i+1}.png'
 if not old.exists() or ImageChops.difference(im.convert('RGB'),Image.open(old).convert('RGB')).getbbox():changed.append(i+1)
 bm.close();pg.close()
data['changed_pages_from_v2']=changed;data['outputs']={p.name:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in [DX,PDF]}
(O/'final_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(data,ensure_ascii=False,indent=2))
