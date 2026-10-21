from load import *
from variance import GROUPS, RUNS
import random, math
random.seed(3)
cors=[]
for it in range(2000):
    xs=[];ys=[]
    games=sorted(RUNS['uc10x'][GROUPS['uc10x'][0]].keys())
    random.shuffle(games); A=games[:12]; B=games[12:24]
    for k,rs in RUNS.items():
        names=list(rs)
        a=[mean(rs[r][g]['score'] for g in A) for r in names]
        b=[mean(rs[r][g]['score'] for g in B) for r in names]
        ma,mb=mean(a),mean(b)
        xs+= [x-ma for x in a]; ys+=[y-mb for y in b]
    num=sum(x*y for x,y in zip(xs,ys)); den=math.sqrt(sum(x*x for x in xs)*sum(y*y for y in ys))
    cors.append(num/den)
cors.sort()
print(f'split-half correlation of run deviations (public-half vs private-half analogue), within config: median {cors[1000]:+.2f}, 10-90% [{cors[200]:+.2f}, {cors[1800]:+.2f}]  (11 runs, 4 groups, 7 dof)')
