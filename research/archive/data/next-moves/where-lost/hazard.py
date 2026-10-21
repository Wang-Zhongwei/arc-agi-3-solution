"""Kaplan-Meier on active minutes per level attempt (all v4 full runs): P(clear in next 10 active min | t min spent)."""
import statistics
from collections import defaultdict
from load import *
from decomp import REFS
ALL = [r for r in REFS if r.startswith('v4-')] + ['v4-level-path-10x/v1']
obs = []  # (active_min, cleared, game, level)
for rel in ALL:
    r = load_run(rel)
    for g in r['games']:
        for lt in level_times(g, r['t_end']):
            obs.append((lt['active'] / 60, lt['cleared'], g['key'], lt['level']))
print('level attempts', len(obs), 'cleared', sum(o[1] for o in obs))
def km(o, grid):
    o = sorted(o)
    out = []
    for t in grid:
        at_risk = [x for x in o if x[0] >= t]
        ev = [x for x in at_risk if x[1] and x[0] < t + 10]
        cens = [x for x in at_risk if (not x[1]) and x[0] < t + 10]
        n = len(at_risk) - len(cens) / 2
        out.append((t, len(at_risk), len(ev), len(ev) / n if n > 0 else float('nan')))
    return out
grid = [0, 5, 10, 15, 20, 25, 30, 40, 50]
print('all levels: t, at-risk, cleared within next 10 min, hazard')
for t, n, e, h in km(obs, grid): print(f"  {t:3d}m n={n:4d} clears={e:4d} P(clear in next 10m)={h:.2f}")
# levels that are usually cleared (rate >= 0.5) vs walls
rate = defaultdict(lambda: [0, 0])
for o in obs:
    rate[(o[2], o[3])][0] += 1; rate[(o[2], o[3])][1] += o[1]
usual = [o for o in obs if rate[(o[2], o[3])][1] / rate[(o[2], o[3])][0] >= 0.5]
wall = [o for o in obs if rate[(o[2], o[3])][1] / rate[(o[2], o[3])][0] < 0.5]
for name, oo in (('usually-cleared levels', usual), ('wall/frontier levels', wall)):
    print(name, len(oo), 'cleared', sum(x[1] for x in oo))
    for t, n, e, h in km(oo, grid): print(f"  {t:3d}m n={n:4d} clears={e:4d} P(clear in next 10m)={h:.2f}")
cl = sorted(o[0] for o in obs if o[1])
print('active minutes to clear (cleared attempts): p50 %.1f p75 %.1f p90 %.1f p95 %.1f' % tuple(cl[int(q * (len(cl) - 1))] for q in (.5, .75, .9, .95)))
