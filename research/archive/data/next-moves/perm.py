from load import *
import statistics as st, math, random
from variance import GROUPS, RUNS
random.seed(7)
def pooled_runsd(assign):
    ss=0;dof=0
    for k,rs in assign.items():
        means=[mean(v.values()) for v in rs]
        m=mean(means); ss+=sum((x-m)**2 for x in means); dof+=len(means)-1
    return math.sqrt(ss/dof)
base={k:[{g:rs[r][g]['score'] for g in rs[r]} for r in rs] for k,rs in RUNS.items()}
obs=pooled_runsd(base)
sims=[]
for _ in range(4000):
    perm={}
    for k,lst in base.items():
        games=list(lst[0]); n=len(lst)
        new=[{} for _ in range(n)]
        for g in games:
            vals=[lst[i][g] for i in range(n)]; random.shuffle(vals)
            for i in range(n): new[i][g]=vals[i]
        perm[k]=new
    sims.append(pooled_runsd(perm))
sims.sort()
print(f'observed pooled run sd {obs:.2f}; under within-run independence: median {sims[2000]:.2f}, 5% {sims[200]:.2f}, 95% {sims[3800]:.2f}; P(sim<=obs)={sum(1 for s in sims if s<=obs)/len(sims):.3f}')
# chi-square CI for pooled sd with dof 7
print('95 pct CI for true run sd from dof 7:',round(obs*math.sqrt(7/16.01),2),round(obs*math.sqrt(7/1.69),2))
