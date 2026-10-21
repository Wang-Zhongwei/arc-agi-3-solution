import sys; sys.path.insert(0,'.')
from cox import get, GROUPS
from load import mean
import math, statistics as st
def lt(cands, refs, key='tok', minL=1):
    diffs=[]
    games=set.intersection(*[set(get(r)) for r in cands+refs])
    for g in games:
        def cost(r,L):
            lv=[l for l in get(r)[g]['levels'] if l['L']==L and l['cleared']]
            return math.log(max(1,lv[0][key] if key!='act' else lv[0]['actions'])) if lv else None
        for L in range(minL,12):
            a=[cost(r,L) for r in cands]; b=[cost(r,L) for r in refs]
            if all(x is not None for x in a+b):
                diffs.append(mean(a)-mean(b))
    m=mean(diffs); se=st.stdev(diffs)/math.sqrt(len(diffs))
    return m,se,len(diffs)
UC=GROUPS['uc10x']; FZ=GROUPS['fz']
tests=[('placebo uc v1',[UC[0]],UC[1:]),('placebo uc v2',[UC[1]],[UC[0],UC[2]]),('placebo uc v3',[UC[2]],UC[:2])]
for i,r in enumerate(FZ): tests.append(('placebo fz '+r,[r],[x for x in FZ if x!=r]))
tests+= [('placebo 13x pair',[GROUPS['13x'][0]],[GROUPS['13x'][1]]),('placebo ledger pair',[GROUPS['ledger'][0]],[GROUPS['ledger'][1]]),
 ('ledger vs fz',GROUPS['ledger'],FZ),('13x vs uc',GROUPS['13x'],UC),('18s-64k vs uc',['v4-uc-18s-64k/v1'],UC),('hats vs uc',['v4-hats/v1'],UC),
 ('stack-v1 vs fz',['v4-stack-v1/v1'],FZ),('stallswap vs fz',['fz-ledger-stallswap/v1'],FZ)]
for key in ('tok','act'):
    zs=[]
    print('== paired log',key,'to clear, levels cleared in all compared runs (negative = cheaper) ==')
    for lab,c,r in tests:
        m,se,n=lt(c,r,key); z=m/se
        if lab.startswith('placebo'): zs.append(z)
        print(f'  {lab:22} mean dlog {m:+.3f} se {se:.3f} z {z:+.2f} n {n}')
    print('  placebo rms z',round(math.sqrt(mean(z*z for z in zs)),2))
