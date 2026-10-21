from load import *
from collections import defaultdict
import math
def attempts(runs):
    out = {}
    for run in runs:
        games, R = load(run)
        byg = defaultdict(list)
        for r in R: byg[r['game']].append(r)
        for g, L in byg.items():
            L.sort(key=lambda r: r['i'])
            bylvl = defaultdict(list)
            for r in L: bylvl[r['lvl']].append(r)
            for lv, X in bylvl.items():
                cleared = any(r['clear'] for r in X)
                if cleared:
                    k = max(j for j, r in enumerate(X) if r['clear']); X = X[:k+1]
                T = sum(r['ct'] for r in X); n = len(X)
                out[(run, g, lv)] = dict(T=T, n=n, MR=sum(r['rt'] for r in X)/n, cleared=cleared,
                                         S4=sum(r['ct'] for r in X if r['rt'] > 4000)/max(1, T),
                                         acts=sum(r['n_act'] for r in X), run=run, g=g, lv=lv)
    return out
import sys
SEL = [r for r in RUNS if not any(x in r for x in sys.argv[1:])]
print("runs:", len(SEL))
A = attempts(SEL)
# within (game, level) fixed effects on cleared attempts
grp = defaultdict(list)
for k, v in A.items():
    if v['cleared'] and v['n'] >= 2: grp[(v['g'], v['lv'])].append(v)
xs, yT, yN, yA = [], [], [], []
for key, V in grp.items():
    if len(V) < 3: continue
    mx = st.mean(math.log(v['MR'] + 50) for v in V)
    mT = st.mean(math.log(v['T']) for v in V); mN = st.mean(math.log(v['n']) for v in V); mA = st.mean(math.log(v['acts'] + 1) for v in V)
    for v in V:
        xs.append(math.log(v['MR'] + 50) - mx); yT.append(math.log(v['T']) - mT); yN.append(math.log(v['n']) - mN); yA.append(math.log(v['acts']+1) - mA)
def slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    res = [c - my - b * (a - mx) for a, c in zip(x, y)]
    se = math.sqrt(sum(e*e for e in res) / (len(x) - 2) / sum((a - mx) ** 2 for a in x))
    return b, se
print(f"cleared level attempts in groups>=3: {len(xs)} attempts, {sum(1 for V in grp.values() if len(V)>=3)} (game,level) groups")
for nm, y in [('log tokens-to-clear', yT), ('log requests-to-clear', yN), ('log actions-to-clear', yA)]:
    b, se = slope(xs, y); print(f"  {nm} on log mean-reasoning-per-request (within game-level): elasticity {b:+.2f} (se {se:.2f})")
# 13x vs 10x matched levels
ten = UC10; thir = ['v4-user-combo/v1', 'v4-user-combo-13x/v1']
def agg(runs):
    d = defaultdict(list)
    for k, v in A.items():
        if v['run'] in runs and v['cleared']: d[(v['g'], v['lv'])].append(v)
    return d
a10, a13 = agg(ten), agg(thir)
common = [k for k in a10 if k in a13]
def gm(xs): return math.exp(st.mean(math.log(x) for x in xs))
r_T = gm([st.mean(v['T'] for v in a13[k]) / st.mean(v['T'] for v in a10[k]) for k in common])
r_n = gm([st.mean(v['n'] for v in a13[k]) / st.mean(v['n'] for v in a10[k]) for k in common])
r_MR = gm([(st.mean(v['MR'] for v in a13[k])+50) / (st.mean(v['MR'] for v in a10[k])+50) for k in common])
r_a = gm([(st.mean(v['acts'] for v in a13[k])+1) / (st.mean(v['acts'] for v in a10[k])+1) for k in common])
print(f"13x vs 10x on {len(common)} levels both cleared: reasoning/request x{r_MR:.2f}, requests-to-clear x{r_n:.2f}, tokens-to-clear x{r_T:.2f}, actions-to-clear x{r_a:.2f}")
# effort ladder natural experiment
import collections
c = collections.Counter()
for run in RUNS:
    for r in load(run)[1]: c[r.get('effort')] += 1
print('reasoning_effort field on requests:', dict(c))
