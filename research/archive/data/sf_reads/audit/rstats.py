import json,glob,re
from collections import Counter,defaultdict
R=json.load(open('SF_rules_labeled.json'))
lev=json.load(open('SF_levels.json'))
# prev-level exposure: requests whose <previous_levels> quotes the rule
pexp=Counter()
for f in sorted(glob.glob('SF/*.req.jsonl')):
    g=f.split('/')[-1][:4]; cur=None
    for l in open(f):
        r=json.loads(l)
        if r.get('pl'): cur=r['pl']
        if not cur: continue
        lvl=None
        for line in cur.split('\n'):
            m=re.match(r'Level (\d+): cleared',line)
            if m: lvl=int(m.group(1))
            if line.startswith('  Rules you confirmed: '):
                for q in re.findall(r'"(.*?)"(?: \| |$)',line[23:]): pexp[(g,lvl,q)]+=1
cls=Counter(); wcls=Counter(); pw=Counter(); contra=0
for d in R:
    cls[d['cls']]+=1; wcls[d['cls']]+=d['n']; d['pexp']=pexp[(d['g'],d['lvl'],d['rule'])]; pw[d['cls']]+=d['pexp']
N=sum(cls.values()); W=sum(wcls.values()); PW=sum(pw.values())
print('classes (distinct, share | this-level turn-weighted share | prev-level request-weighted share)')
for c in 'MTPHXE': print(c, cls[c], '%.0f%%'%(100*cls[c]/N), '| %.0f%%'%(100*wcls[c]/W), '| %.0f%%'%(100*pw[c]/PW))
print('total this-level rule-impressions',W,'prev-level rule-impressions',PW)
print('contradicted flagged',sum(d['contra'] for d in R), 'their this-level impressions', sum(d['n'] for d in R if d['contra']))
# fragments / question marks / coordinates
frag=[d for d in R if d['rule'][:1].islower() or d['rule'][:1] in ')]' or d['rule'].rstrip().endswith((':','...','…'))]
print('fragments (starts lowercase/")" or ends ":"/"..."):',len(frag))
qm=[d for d in R if '?' in d['rule']]; print('contain "?":',len(qm))
coord=[d for d in R if re.search(r'\(\s*\d+\s*,\s*\d+\s*\)|rows? \d+|cols? \d+|col \d+',d['rule'])]; print('with coordinates/rows/cols:',len(coord), Counter(d['cls'] for d in coord))
echo=[d for d in R if re.search(r'level[- ]\d|levels \d',d['rule'],re.I)]; print('mention another level:',len(echo),Counter(d['cls'] for d in echo))
print('sentence opens with So:',sum(1 for d in R if d['rule'].startswith('So')), Counter(d['cls'] for d in R if d['rule'].startswith('So')))
mark=[d for d in R if re.search(r'[✓✔✅]|\bconfirm|\bverified',d['rule'],re.I)]; print('carry a mark:',len(mark),Counter(d['cls'] for d in mark))
nomark=[d for d in R if d not in mark]; print('no mark (inherited conclusions):',len(nomark),Counter(d['cls'] for d in nomark))
# per game non-M share weighted
pg=defaultdict(lambda:[0,0])
for d in R: pg[d['g']][0]+=d['n']*(d['cls']!='M'); pg[d['g']][1]+=d['n']
print('per-game non-mechanic share of this-level rule impressions (worst first):')
print(sorted(((round(a/b,2) if b else 0,g,b) for g,(a,b) in pg.items()),reverse=True))
# stall levels: >=60k completion tokens
stall={(g,int(k)) for g,v in lev.items() for k,d in v.items() if d['ctok']>=60000}
sw=Counter(); nw=Counter()
for d in R:
    (sw if (d['g'],d['lvl']) in stall else nw)[d['cls']]+=d['n']
print('in stalled levels (>=60k tok):',{c:'%.0f%%'%(100*sw[c]/sum(sw.values())) for c in 'MTPHXE'}, sum(sw.values()))
print('in other levels:',{c:'%.0f%%'%(100*nw[c]/sum(nw.values())) for c in 'MTPHXE'}, sum(nw.values()))
json.dump(R,open('SF_rules_labeled.json','w'),indent=0,ensure_ascii=False)
