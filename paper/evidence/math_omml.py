"""Small strict LaTeX -> MathML -> editable Office Math converter.

Uses only the bundled lxml and the Microsoft Office MML2OMML stylesheet.
Public interface: to_omml(latex: str) -> lxml.etree._Element (m:oMath).
Unsupported commands or malformed groups raise LatexSyntaxError; nothing is
silently discarded. This is intentionally a subset parser, not general TeX.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
from lxml import etree

MML = 'http://www.w3.org/1998/Math/MathML'
OMML = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
XSL_PATH = Path(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL')
_TRANSFORM = None

class LatexSyntaxError(ValueError):
    pass

def node(tag, *children, text=None, **attrs):
    el = etree.Element('{%s}%s' % (MML, tag), **{k.rstrip('_'): str(v) for k,v in attrs.items()})
    if text is not None: el.text = text
    for child in children: el.append(child)
    return el

def row(children):
    if len(children) == 1: return children[0]
    return node('mrow', *children)

GREEK = {
    'alpha':'α','beta':'β','gamma':'γ','delta':'δ','epsilon':'ϵ','varepsilon':'ε',
    'zeta':'ζ','eta':'η','theta':'θ','vartheta':'ϑ','iota':'ι','kappa':'κ',
    'lambda':'λ','mu':'μ','nu':'ν','xi':'ξ','pi':'π','rho':'ρ','sigma':'σ',
    'tau':'τ','upsilon':'υ','phi':'ϕ','varphi':'φ','chi':'χ','psi':'ψ','omega':'ω',
    'Gamma':'Γ','Delta':'Δ','Theta':'Θ','Lambda':'Λ','Xi':'Ξ','Pi':'Π',
    'Sigma':'Σ','Upsilon':'Υ','Phi':'Φ','Psi':'Ψ','Omega':'Ω'
}
OPERATORS = {
    'times':'×','cdot':'·','in':'∈','notin':'∉','le':'≤','leq':'≤','ge':'≥','geq':'≥',
    'ne':'≠','neq':'≠','approx':'≈','equiv':'≡','to':'→','rightarrow':'→',
    'leftarrow':'←','Rightarrow':'⇒','mapsto':'↦','pm':'±','mp':'∓','cup':'∪',
    'cap':'∩','subset':'⊂','subseteq':'⊆','emptyset':'∅','infty':'∞',
    'forall':'∀','exists':'∃','partial':'∂','nabla':'∇'
}
FUNCTIONS = {'log','ln','exp','max','min','arg','sin','cos','tan','tanh','sigmoid','det','lim'}
SPACES = {'quad':'1em','qquad':'2em',',':'0.1667em',';':'0.2778em',':':'0.2222em','!':'-0.1667em',' ':'0.3333em'}
STYLES = {'mathrm':'normal','mathit':'italic','mathbf':'bold','mathcal':'script','mathbb':'double-struck','mathsf':'sans-serif','mathtt':'monospace'}
DELIMS = {'(' : ')','[':']'}
TOKEN_RE = re.compile(r'\\[A-Za-z]+|\\.|\d+(?:\.\d+)?|[A-Za-z]|[^\s]',re.UNICODE)

class Parser:
    def __init__(self, text):
        self.source=text
        self.tokens=TOKEN_RE.findall(text)
        self.i=0
    def peek(self): return self.tokens[self.i] if self.i<len(self.tokens) else None
    def pop(self):
        token=self.peek()
        if token is None: raise LatexSyntaxError('Unexpected end of equation')
        self.i+=1
        return token
    def expect(self, token):
        got=self.pop()
        if got!=token: raise LatexSyntaxError(f'Expected {token!r}, found {got!r} at token {self.i}')
    def expression(self, stop=None):
        children=[]
        while self.peek() is not None and self.peek()!=stop:
            if self.peek() in {'}',')',']',r'\right'}:
                raise LatexSyntaxError(f'Unexpected closing delimiter {self.peek()!r}')
            children.append(self.atom())
        if stop is not None and self.peek()!=stop:
            raise LatexSyntaxError(f'Unclosed group, expected {stop!r}')
        return row(children)
    def argument(self):
        if self.peek()=='{':
            self.pop(); value=self.expression('}'); self.expect('}'); return value
        # A TeX unbraced argument is one primary, not a primary with its scripts.
        return self.primary()
    def plain_group(self):
        self.expect('{'); parts=[]
        while self.peek() is not None and self.peek()!='}':
            t=self.pop()
            if t.startswith('\\') or t=='{':
                raise LatexSyntaxError('Nested commands not supported in operatorname/text')
            parts.append(t)
        self.expect('}')
        return ''.join(parts)
    def atom(self):
        base=self.primary(); sub=None; sup=None
        while self.peek() in {'_','^'}:
            kind=self.pop(); value=self.argument()
            if kind=='_':
                if sub is not None: raise LatexSyntaxError('Duplicate subscript')
                sub=value
            else:
                if sup is not None: raise LatexSyntaxError('Duplicate superscript')
                sup=value
        if sub is None and sup is None: return base
        is_nary = base.tag=='{%s}mo'%MML and base.text in {'∑','∏','∫','⋃','⋂'}
        if sub is not None and sup is not None:
            return node('munderover' if is_nary else 'msubsup',base,sub,sup)
        if sub is not None: return node('munder' if is_nary else 'msub',base,sub)
        return node('mover' if is_nary else 'msup',base,sup)
    def primary(self):
        t=self.pop()
        if t=='{':
            value=self.expression('}');self.expect('}');return value
        if t in DELIMS:
            close=DELIMS[t];inside=self.expression(close);self.expect(close)
            return node('mrow',node('mo',text=t,fence='true',stretchy='true'),inside,node('mo',text=close,fence='true',stretchy='true'))
        if t.startswith('\\'):
            cmd=t[1:]
            if cmd in GREEK:return node('mi',text=GREEK[cmd])
            if cmd in OPERATORS:return node('mo',text=OPERATORS[cmd])
            if cmd in FUNCTIONS:return node('mi',text=cmd,mathvariant='normal')
            if cmd in SPACES:return node('mspace',width=SPACES[cmd])
            if cmd in {'frac','dfrac','tfrac'}:return node('mfrac',self.argument(),self.argument())
            if cmd=='sqrt':
                if self.peek()=='[':
                    self.pop(); index=self.expression(']');self.expect(']')
                    return node('mroot',self.argument(),index)
                return node('msqrt',self.argument())
            if cmd in {'bar','overline','hat','widehat','tilde','widetilde','vec','dot','ddot'}:
                accent={'bar':'¯','overline':'¯','hat':'^','widehat':'^','tilde':'˜','widetilde':'˜','vec':'→','dot':'˙','ddot':'¨'}[cmd]
                return node('mover',self.argument(),node('mo',text=accent,stretchy='true'),accent='true')
            if cmd in STYLES:
                value=self.argument()
                for el in value.iter():
                    if el.tag in {'{%s}mi'%MML,'{%s}mn'%MML}:el.set('mathvariant',STYLES[cmd])
                return value
            if cmd in {'operatorname','text'}:
                return node('mi' if cmd=='operatorname' else 'mtext',text=self.plain_group(),mathvariant='normal')
            if cmd in {'sum','prod','int','bigcup','bigcap'}:
                return node('mo',text={'sum':'∑','prod':'∏','int':'∫','bigcup':'⋃','bigcap':'⋂'}[cmd],largeop='true',movablelimits='false')
            if cmd=='left':
                opening=self.pop()
                if opening not in {'(','[','|',r'\{','.'}:raise LatexSyntaxError(f'Unsupported left delimiter {opening!r}')
                inside=self.expression(r'\right');self.expect(r'\right');closing=self.pop()
                if closing not in {')',']','|',r'\}','.'}:raise LatexSyntaxError(f'Unsupported right delimiter {closing!r}')
                out=[]
                if opening!='.':out.append(node('mo',text=opening.lstrip('\\'),fence='true',stretchy='true'))
                out.append(inside)
                if closing!='.':out.append(node('mo',text=closing.lstrip('\\'),fence='true',stretchy='true'))
                return node('mrow',*out)
            if cmd in {'{','}','%','#','&','_'}:return node('mo',text=cmd)
            raise LatexSyntaxError(f'Unsupported LaTeX command \\{cmd} at token {self.i}')
        if re.fullmatch(r'\d+(?:\.\d+)?',t):return node('mn',text=t)
        if re.fullmatch(r'[A-Za-z]',t):return node('mi',text=t)
        if t in {'_','^','}',')',']','$'}:raise LatexSyntaxError(f'Unexpected token {t!r}')
        if t in '=+-*/,:;|<>!':return node('mo',text='−' if t=='-' else t)
        if t.isalpha():return node('mi',text=t)
        raise LatexSyntaxError(f'Unsupported literal token {t!r}')

def to_mathml(latex):
    parser=Parser(latex)
    result=parser.expression()
    if parser.peek() is not None:raise LatexSyntaxError(f'Trailing tokens at {parser.i}')
    math=etree.Element('{%s}math'%MML,nsmap={None:MML},display='block')
    math.append(result)
    return math

def to_omml(latex):
    global _TRANSFORM
    if _TRANSFORM is None:
        if not XSL_PATH.is_file():raise FileNotFoundError(XSL_PATH)
        _TRANSFORM=etree.XSLT(etree.parse(str(XSL_PATH)))
    result=_TRANSFORM(to_mathml(latex))
    root=result.getroot()
    if root is None:raise ValueError('MathML transform returned no result')
    if root.tag!='{%s}oMath'%OMML:
        candidates=root.findall('.//{%s}oMath'%OMML)
        if len(candidates)!=1:raise ValueError('Expected exactly one m:oMath')
        root=candidates[0]
    if not ''.join(root.itertext()).strip():raise ValueError('Math conversion produced no visible content')
    return deepcopy(root)

if __name__=='__main__':
    import json,sys
    from docx import Document
    from docx.shared import Pt
    sys.stdout.reconfigure(encoding='utf-8')
    here=Path(__file__).parent
    equations=[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('type')=='equation':equations.append(x)
            for value in x.values():walk(value)
        elif isinstance(x,list):
            for value in x:walk(value)
    for name in ('chapters_1_2.json','chapters_3_4.json'):
        walk(json.loads((here/name).read_text(encoding='utf-8')))
    doc=Document()
    doc.styles['Normal'].font.name='Cambria Math'
    doc.styles['Normal'].font.size=Pt(12)
    evidence=[]
    for e in equations:
        doc.add_paragraph('Equation '+e['label'])
        parts=re.split(r',\\qquad\s*|,\\quad\s*',e['latex'])
        all_text=[]
        for part in parts:
            el=to_omml(part)
            all_text.append(''.join(el.itertext()))
            p=doc.add_paragraph();p._p.append(el)
        evidence.append({'label':e['label'],'parts':len(parts),'visible_text':all_text})
    for invalid in [r'\unknown{x}',r'\frac{1}',r'x_{a',r'x^^2']:
        try:to_omml(invalid)
        except LatexSyntaxError:pass
        else:raise AssertionError('Unknown or malformed input was accepted: '+invalid)
    doc.save(here/'math_validation.docx')
    (here/'math_validation.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS:',len(equations),'equations;',sum(v['parts'] for v in evidence),'editable math parts; malformed commands rejected')
