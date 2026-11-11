from pathlib import Path
import sys, os, importlib.util, shutil
SKILL=Path(r'C:\Users\ROBOT\.codex\plugins\cache\openai-primary-runtime\documents\26.1104.22356\skills\documents')
POP=Path(r'C:\Users\ROBOT\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')
os.environ['PATH']=str(POP)+os.pathsep+os.environ['PATH']
spec=importlib.util.spec_from_file_location('packaged_render',SKILL/'render_docx.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
pdf=Path(sys.argv.pop(1)).resolve()
def native_pdf(*args,**kwargs): return str(pdf),'Microsoft Word 16 native PDF; original source read-only; no LibreOffice used.'
mod.convert_to_pdf=native_pdf
mod.main()
