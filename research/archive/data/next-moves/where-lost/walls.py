"""Per-game category means over the 3 user-combo-10x runs, and the 'wall' table: run-points if one level's
empirical clear rate (given reached) went to 1, from the per-level ladder of all v4 full runs."""
import json, statistics
from collections import defaultdict
from load import *
from decomp import REFS, decomp_game, ref_clear_times

TEN = ['v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers']
need, reached = ref_clear_times()
import decomp; decomp.REACHED.update(reached)
per = defaultdict(lambda: defaultdict(float))
for rel in TEN:
    r = load_run(rel)
    for g in r['games']:
        x = decomp_game(g, r['t_end'], need)
        p = per[x['game']]
        p['S'] += x['S'] / 3; p['ineff'] += x['ineff'] / 3; p['d_fail'] += x['d_fail'] / 3
        cls = x['cls'] if x['cls'] != 'not_started' else 'clock'
        if cls != 'won':
            p[cls] += (x['stuck_w'] + x['beyond_w']) / 3
            p[cls + '_next'] += x['stuck_w'] / 3
print('per game, mean of 3 user-combo-10x runs (game points; divide by 25 for run points)')
print(f"{'game':5s} {'score':>6s} {'c_ineff':>7s} {'(d_fail)':>8s} {'clock':>6s} {'stuck':>6s} {'wall':>6s} {'frontier':>8s}   next-level-only: clock/stuck/wall/frontier")
tot = defaultdict(float)
for k, p in sorted(per.items(), key=lambda kv: kv[1]['S']):
    print(f"{k:5s} {p['S']:6.1f} {p['ineff']:7.1f} {p['d_fail']:8.1f} {p['clock']:6.1f} {p['stuck']:6.1f} {p['wall']:6.1f} {p['frontier']:8.1f}   {p['clock_next']:5.1f}/{p['stuck_next']:5.1f}/{p['wall_next']:5.1f}/{p['frontier_next']:5.1f}")
    for kk, v in p.items(): tot[kk] += v / 25
print('run points:', {k: round(v, 2) for k, v in sorted(tot.items())})

# wall table over all v4 full runs
ALL = [r for r in REFS if r.startswith('v4-')] + ['v4-level-path-10x/v1']
lad = defaultdict(lambda: defaultdict(lambda: [0, 0]))
nlv = {}
for rel in ALL:
    r = load_run(rel)
    for g in r['games']:
        nlv[g['key']] = g['nl']
        if not g['at']:
            lad[g['key']][0][0] += 1; continue
        for i in range(min(g['lc'] + 1, g['nl'])):
            lad[g['key']][i][0] += 1
            if i < g['lc']: lad[g['key']][i][1] += 1
def expected(k, override=None):
    nl = nlv[k]; W = nl * (nl + 1) / 2; e = 0.0; pr = 1.0
    for i in range(nl):
        a, c = lad[k][i]
        p = (c / a) if a else 0.0
        if override is not None and i == override: p = 1.0
        pr *= p; e += 100 * (i + 1) / W * pr
    return e
print('\nwalls (all %d v4 full runs): level, cleared/reached, run-points if that level always cleared once reached' % len(ALL))
walls = []
for k in nlv:
    base = expected(k)
    for i in range(nlv[k]):
        a, c = lad[k][i]
        if a == 0 or c == a: continue
        walls.append((expected(k, i) - base) / 25 if True else 0)
        walls[-1] = (walls[-1], k, i, c, a)
print('expected score from ladder product:', round(sum(expected(k) for k in nlv) / 25, 2))
for v, k, i, c, a in sorted(walls, reverse=True)[:25]:
    print(f"  {k} L{i+1}: {c}/{a} cleared  -> +{v:.2f} run-points")
