"""Level-stratified time-to-clear comparison (Gehan pairwise, censored), permutation p-value.
Each (game, level) reached in both runs is a stratum. A 'wins' if it cleared and the other run either cleared later or
was still uncleared after more active time. Indeterminate pairs score 0. Active time = min(gap, tokens/rate_run)."""
import random, statistics, sys, itertools
import load
from load import *

def run_rate(r):
    rs = []
    for g in r['games']:
        prev = g['off']
        for t, tk in zip(g['at'], g['tokens']):
            gap = t - prev; prev = t
            if 5 < gap <= 240: rs.append(tk / gap)
    return statistics.median(rs)

CACHE = {}
def attempts(rel):
    if rel in CACHE: return CACHE[rel]
    r = load_run(rel); load.RATE = run_rate(r)
    d = {}
    for g in r['games']:
        for lt in level_times(g, r['t_end']):
            d[(g['key'], lt['level'])] = (lt['active'] / 60, lt['cleared'])
    load.RATE = 55.0
    CACHE[rel] = d; return d

def gehan(a, b):
    (ta, ca), (tb, cb) = a, b
    if ca and (not cb or tb > ta) and (cb or tb >= ta): return 1
    if cb and (not ca or ta > tb) and (ca or ta >= tb): return -1
    return 0

def compare(A, B, nperm=4000, seed=0):
    pairs = []  # list of strata scores, one per (runA, runB, stratum)
    blocks = {}  # stratum -> summed score over run pairs; signs flip per stratum (conservative block permutation)
    for ra, rb in itertools.product(A, B):
        da, db = attempts(ra), attempts(rb)
        for k in set(da) & set(db):
            s_ = gehan(da[k], db[k]); pairs.append(s_); blocks[k] = blocks.get(k, 0) + s_
    U = sum(pairs); dec = sum(1 for p in pairs if p != 0)
    bl = list(blocks.values())
    rnd = random.Random(seed); cnt = 0
    for _ in range(nperm):
        u = sum(p if rnd.random() < 0.5 else -p for p in bl)
        if abs(u) >= abs(U): cnt += 1
    return U, dec, len(pairs), (sum(1 for p in pairs if p > 0) / max(1, dec)), cnt / nperm

if __name__ == '__main__':
    TEN = ['v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers']
    tests = [('placebo 10x v1 vs v2', [TEN[0]], [TEN[1]]), ('placebo 10x v1 vs v3', [TEN[0]], [TEN[2]]), ('placebo 10x v2 vs v3', [TEN[1]], [TEN[2]]),
             ('10x(3) vs hats (37.11)', TEN, ['v4-hats/v1']), ('10x(3) vs 16x (39.65)', TEN, ['v4-user-combo-16x/v1']),
             ('10x(3) vs 18s-64k (36.01)', TEN, ['v4-uc-18s-64k/v1']), ('10x(3) vs 13x (47.34)', TEN, ['v4-user-combo-13x/v1']),
             ('10x(3) vs accept05 (46.77)', TEN, ['v4-user-combo-10x-accept05/v1']), ('10x(3) vs uc-11x (47.19)', TEN, ['v4-uc-11x/v1-snowiers']),
             ('10x(3) vs curriculum (47.83)', TEN, ['v4-sched-curriculum/v1']), ('10x(3) vs level-path (48.31)', TEN, ['v4-level-path-10x/v1']),
             ('10x(3) vs v4-control (50.62)', TEN, ['v4-control/v1']), ('10x(3) vs stack-v1 (43.32)', TEN, ['v4-stack-v1/v1']),
             ('10x v1 alone vs hats', [TEN[0]], ['v4-hats/v1']), ('10x v3 alone vs hats', [TEN[2]], ['v4-hats/v1']),
             ('10x v3 alone vs 18s-64k', [TEN[2]], ['v4-uc-18s-64k/v1'])]
    for name, A, B in tests:
        U, dec, n, fa, p = compare(A, B)
        print(f"{name:34s} strata-pairs {n:4d} decided {dec:4d}  A faster in {fa:.0%} of decided  U={U:+4d}  perm p={p:.3f}")
