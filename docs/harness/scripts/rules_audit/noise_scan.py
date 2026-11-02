import json,sys,re,collections as C
from audit_paths import DATA
import rule_v8 as R
rows=[r for r in json.load(open(DATA/'joined_v02.json')) if r['kind']=='confirmed' and r['in_section'] and r.get('claim_verdict') and r.get('same') and r.get('later')]
ok=lambda r:r['claim_verdict']=='correct' or (r['claim_verdict']=='partially_correct' and r['usefulness']!='misleading')
bad=lambda r:r['claim_verdict']=='incorrect' or r['usefulness']=='misleading'
useful=lambda r,rd:ok(r) and r[rd]['meaningful'] and not bad(r)
COORD=re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)|\brows?\s*\d+|\bcols?\s*\d+|\bcolumns?\s*\d+|\b[rc]\d+\b")
GEN=re.compile(r"\b(every|each|always|never|any|whenever|when|after|per|causes?|makes?|means?|so|because|toggles?|moves?|kills?|blocks?|resets?|costs?|adds?|removes?|changes?|turns?|teleports?|pushes|swaps?|triggers?|requires?|wins?|must|only|rule|mechanic)\b",re.I)
SNAP_W=re.compile(r"\b(now|still|stayed|currently|at this step|fresh state|state confirmed|initial|at steps?|present|remaining)\b",re.I)
def body(s): return re.sub(r"\bconfirmed\b|✓|✗","",s,flags=re.I)
def nw(s): return len(body(s).split())
def F(r):
    s=r['shown']; c=len(COORD.findall(s)); w=nw(s); g=bool(GEN.search(body(s)))
    return {
    'colon_end':s.rstrip().endswith(':'),
    'short<=6':w<=6,
    'short<=8':w<=8,
    'step_ref':bool(re.search(r"^\W*(step\s*\d|then\b)|\bsteps?\s*\d",s,re.I)),
    'ref_elsewhere':bool(re.search(r"confirmed (earlier|before|above|in level|it\b|that\b)|\[confirmed|level \d+ confirmed|confirmed that\.?$",s,re.I)),
    'coord>=1':c>=1,
    'coord>=2':c>=2,
    'coord>=3':c>=3,
    'coord>=1&nogen':c>=1 and not g,
    'coord>=2&nogen':c>=2 and not g,
    'coord>=3&nogen':c>=3 and not g,
    'snapword&coord':bool(SNAP_W.search(s)) and c>=1,
    'snapword&coord&nogen':bool(SNAP_W.search(s)) and c>=1 and not g,
    'nogen':not g,
    'nogen&short<=10':not g and w<=10,
    'digits>25%':sum(ch.isdigit() for ch in s)/max(1,len(s))>.25,
    'digits>15%':sum(ch.isdigit() for ch in s)/max(1,len(s))>.15,
    'digits>10%':sum(ch.isdigit() for ch in s)/max(1,len(s))>.10,
    'confirmed_lead+coord':bool(re.match(r"\W*confirmed\b",s,re.I)) and c>=1,
    'also_lead':bool(re.match(r"\W*(also )?confirmed:",s,re.I)),
    }
for ph,uni,rd in [('current',rows,'same'),('past',[r for r in rows if r['level']<r['max_level_seen']],'later')]:
    k=[r for r in uni if R.keep(r['shown'],r['next1'],ph)]
    tot=C.Counter('bad' if bad(r) else 'useful' if useful(r,rd) else 'noise' for r in k)
    print('\n##',ph,len(k),dict(tot))
    print('%-24s %4s %4s %4s %4s  %%noise+bad'%('feature','n','use','bad','noise'))
    res={}
    for r in k:
        for f,v in F(r).items():
            if v:
                res.setdefault(f,C.Counter())['bad' if bad(r) else 'useful' if useful(r,rd) else 'noise']+=1
    for f,c in sorted(res.items(),key=lambda x:-(x[1]['noise']+x[1]['bad'])/max(1,sum(x[1].values()))):
        n=sum(c.values())
        print('%-24s %4d %4d %4d %4d  %3.0f%%'%(f,n,c['useful'],c['bad'],c['noise'],100*(c['noise']+c['bad'])/n))
