from load import *
from collections import defaultdict
for name, runs in {'uc10x (3 runs)': UC10, '13x (2 runs)': ['v4-user-combo/v1','v4-user-combo-13x/v1'], 'all 14 runs': RUNS}.items():
    R=[]
    for run in runs: R+=load(run)[1]
    G=sum(r['ct'] for r in R); RE=sum(r['rt'] for r in R)
    print(f"\n== {name}: {len(R)} req, gen {G/1e6:.2f}M, reasoning {100*RE/G:.1f}% ==")
    B=[(0,2000),(2000,4000),(4000,6000),(6000,8000),(8000,12000),(12000,99999)]
    print(f"{'bin':<12}{'req%':>6}{'gen%':>6}{'acts/req':>9}{'clear in req %':>15}{'clear<=3 req %':>15}")
    byg=defaultdict(list)
    for r in R: byg[(r['run'],r['game'])].append(r)
    for L in byg.values():
        L.sort(key=lambda r:r['i'])
        cl=[j for j,r in enumerate(L) if r['clear']]
        for j,r in enumerate(L): r['c3']=any(j<=k<j+3 for k in cl)
    for lo,hi in B:
        X=[r for r in R if lo<=r['rt']<hi]
        if not X: continue
        print(f"{lo:>5}-{hi:<6}{100*len(X)/len(R):6.2f}{100*sum(r['ct'] for r in X)/G:6.1f}{sum(r['n_act'] for r in X)/len(X):9.2f}{100*sum(1 for r in X if r['clear'])/len(X):15.2f}{100*sum(r['c3'] for r in X)/len(X):15.1f}")
    print('budget  gen-tokens-saved%(upper bound)  requests-cut%  clearing-requests-cut%  cut-requests-that-were-12k-dead%')
    clr=[r for r in R if r['clear']]
    for cap in (3000,4000,6000,8000,10000):
        sav=sum(max(0,r['rt']-cap) for r in R)
        cut=[r for r in R if r['rt']>cap]
        print(f"{cap:>6}  {100*sav/G:8.1f}  {100*len(cut)/len(R):8.2f}  {100*sum(1 for r in clr if r['rt']>cap)/len(clr):8.2f}  {100*sum(1 for r in cut if r['fin']=='length')/max(1,len(cut)):8.1f}")
