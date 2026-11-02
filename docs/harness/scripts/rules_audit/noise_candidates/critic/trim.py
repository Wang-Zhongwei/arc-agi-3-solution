import sys,re,importlib.util,collections as C
sys.path.insert(0,'/Users/wang/.claude/jobs/910e124b/tmp/scripts'); sys.path.insert(0,'critic')
import noise_harness as H, try1 as T, adv as A
V={p:H.views(p) for p in H.PH}
def sc(fn,ph):
    hit=[r for r,l in V[ph] if fn(l)]; c=C.Counter(H.lab(r,H.PH[ph]) for r in hit)
    return len(hit),c['useful'],c['bad'],c['noise'],len({r['game'] for r in hit})
ALT=['→','when','whenever','if','while','after','until','per','each','every','always','never','so','by']
INP=['click','press','mov','push','drag']
def mk(alts,inp):
    w=[a for a in alts if a!='→']
    pat=('→|' if '→' in alts else '')+r"\b("+'|'.join(w)+r")\b"
    pat2=r"\b("+'|'.join(inp)+r")\w*" if inp else r"(?!x)x"
    g=re.compile(pat,re.I); i=re.compile(pat2,re.I)
    T.active=lambda t:bool(g.search(t) or i.search(t))
def rep(lbl):
    cur=T.CANDIDATES['U_cur'][1]; pa=T.CANDIDATES['U_past_a'][1]; ad=sum(1 for s,_ in A.ADV if pa(A.line(s))); adc=sum(1 for s,_ in A.ADV if cur(A.line(s)))
    print(f"{lbl:22s} cur {sc(cur,'current')} past {sc(pa,'past')} adv cur {adc} past {ad}")
mk(ALT,INP); rep('all')
for a in ALT:
    mk([x for x in ALT if x!=a],INP); rep('-'+a)
for a in INP:
    mk(ALT,[x for x in INP if x!=a]); rep('-'+a)
mk(['→','when','if','after','per','each','every','always'],['click','press','mov','push','drag']); rep('small8+5')
mk(['→','when','if','per','each','every','always'],['click','press','mov','push','drag']); rep('small7+5')
mk(['→','when','if','after','per','each','every','always'],[]); rep('small8 no input')
mk(['→','when','whenever','if','after','per','each','every','always','by'],['click','press','mov','push','drag']); rep('small10+5')
