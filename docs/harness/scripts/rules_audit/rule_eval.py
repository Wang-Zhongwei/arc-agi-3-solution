import json, re, sys, itertools, collections as C
from audit_paths import DATA
V = f'{DATA}/'
rows=json.load(open(sys.argv[1] if len(sys.argv)>1 else V+'cold_v02.json'))
conf=[r for r in rows if r['kind']=='confirmed']
HEDGE=re.compile(r"^\W*(hmm+|wait|but|however|unless|actually|although|though|maybe|perhaps|or\b|not sure|hold on|let me (re-?check|double|verify|re-?read)|is that|did\b|does\b|was\b|could|might|need to (check|verify))|\?\s*$|\b(not sure|unclear|double[- ]check|re-?check|verify|uncertain|suspicious|contradict|doesn't (match|fit)|wrong)\b", re.I)
COORD=re.compile(r'\(\s*-?\d+\s*,\s*-?\d+\s*\)|\b(rows?|cols?|columns?)\s*\d+', re.I)
IMPOSS=re.compile(r"unsolvable|impossib|dead end|\btrap\b|no exit|no way|can't be (reached|solved)|cannot be (reached|solved)|must be wrong|is wrong", re.I)
META=re.compile(r"^\W*(so|ok|okay|good|great|yes|and|also|no,?)?[,\s]*(the|this|that|my|our|it|which|we|i('ve)?)\b[^:]{0,40}?\b(confirmed|verified)\b[^:]{0,50}$", re.I)
def content_words(s):
    STOP=set('the a an is are was were be been to of in on at and or it its this that these those by with for as from so not no confirmed verified also my our we i ive have has had all now both again earlier empirically twice exactly fully already ok okay good great yes'.split())
    return [w for w in re.findall(r"[a-z][a-z'-]+",s.lower()) if w not in STOP]
def meta(s): return bool(META.search(s)) and len(content_words(s))<=4
feat=lambda r: dict(H=bool(HEDGE.search(r['next1'] or '')), A=('CONFIRMED' in r['shown'] or r['shown'].count('✓')>=2), M=meta(r['shown']),
                    C1=len(COORD.findall(r['shown']))>=1, C2=len(COORD.findall(r['shown']))>=2, I=bool(IMPOSS.search(r['shown'])),
                    cleared=r['level']<r['max_level_seen'])
for r in conf: r['F']=feat(r)
ok=lambda r: r['claim_verdict']=='correct' or (r['claim_verdict']=='partially_correct' and r['usefulness']!='misleading')
bad=lambda r: r['claim_verdict']=='incorrect' or r['usefulness']=='misleading'
def score(lines, reader):
    use=[r for r in lines if ok(r) and r[reader]['meaningful'] and not bad(r)]
    b=[r for r in lines if bad(r)]
    n=len(lines); return n,len(use),len(b),n-len(use)-len(b)
def show(title, universe, reader, rules):
    tot=score(universe,reader)
    print('\n## %s  (universe %d lines, useful %d, bad %d, noise %d)'%(title,)+tot if False else '\n## %s  (universe n=%d: useful %d, bad %d, noise %d)'%((title,)+tot))
    print('   %-44s %5s %7s %7s %7s %9s %8s'%('rule','kept','useful','bad','noise','precision','recall'))
    for name,keep in rules:
        k=[r for r in universe if keep(r['F'])]; n,u,b,z=score(k,reader)
        print('   %-44s %5d %7d %7d %7d %8.0f%% %7.0f%%'%(name,n,u,b,z,100*u/max(1,n),100*u/max(1,tot[1])))
RULES=[('keep all',lambda f:True),
 ('drop hedge-after (H)',lambda f:not f['H']),
 ('H + meta (M)',lambda f:not f['H'] and not f['M']),
 ('H + M + agitated (A)',lambda f:not(f['H'] or f['M'] or f['A'])),
 ('H + M + impossibility (I)',lambda f:not(f['H'] or f['M'] or f['I'])),
 ('H + M + A + I',lambda f:not(f['H'] or f['M'] or f['A'] or f['I'])),
 ('H + M + coords>=2 (C2)',lambda f:not(f['H'] or f['M'] or f['C2'])),
 ('H + M + coords>=1 (C1)',lambda f:not(f['H'] or f['M'] or f['C1'])),
 ('H + M + A + I + C2',lambda f:not(f['H'] or f['M'] or f['A'] or f['I'] or f['C2'])),
 ('H + M + A + I + C1',lambda f:not(f['H'] or f['M'] or f['A'] or f['I'] or f['C1'])),
]
past=[r for r in conf if r['F']['cleared']]
show('PAST block: claims from cleared levels, judged by the later-level reader', past, 'later', RULES)
show('CURRENT block: all claims while on their own level, judged by the same-level reader', conf, 'same', RULES)
show('v7 TODAY: every claim at every later level (later reader)', conf, 'later', RULES[:2])
