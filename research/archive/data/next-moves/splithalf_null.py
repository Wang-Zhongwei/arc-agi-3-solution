from load import *
from variance import GROUPS, RUNS
import random, math
random.seed(5)
base={k:[{g:rs[r][g]['score'] for g in rs[r]} for r in rs] for k,rs in RUNS.items()}
games0=sorted(base['uc10x'][0].keys())
def sh(data,splits=200):
    cs=[]
    for _ in range(splits):
        games=games0[:]; random.shuffle(games); A=games[:12]; B=games[12:24]
        xs=[];ys=[]
        for k,lst in data.items():
            a=[mean(r[g] for g in A) for r in lst]; b=[mean(r[g] for g in B) for r in lst]
            ma,mb=mean(a),mean(b); xs+=[x-ma for x in a]; ys+=[y-mb for y in b]
        cs.append(sum(x*y for x,y in zip(xs,ys))/math.sqrt(sum(x*x for x in xs)*sum(y*y for y in ys)))
    cs.sort(); return cs[len(cs)//2]
obs=sh(base,1000)
nulls=[]
for _ in range(400):
    perm={}
    for k,lst in base.items():
        n=len(lst); new=[{} for _ in range(n)]
        for g in games0:
            vals=[lst[i][g] for i in range(n)]; random.shuffle(vals)
            for i in range(n): new[i][g]=vals[i]
        perm[k]=new
    nulls.append(sh(perm,40))
nulls.sort()
print(f'observed median split-half r {obs:+.2f}; null (games independent within run) median {nulls[200]:+.2f}, 5% {nulls[20]:+.2f}, 95% {nulls[380]:+.2f}; P(null<=obs) {sum(1 for x in nulls if x<=obs)/len(nulls):.3f}')
