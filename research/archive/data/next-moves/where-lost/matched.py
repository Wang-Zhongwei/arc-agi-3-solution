"""Matched wall-clock (benchmark seconds) comparison: score, cleared-weight cap, inefficiency, failed-attempt actions."""
import sys
from load import *

def at_time(g, T):
    if g['off'] is None or not g['at']:
        return 0, [0] * g['nl'], 0
    n = sum(1 for t in g['at'] if t <= T)
    lc = 0; apl = [0] * g['nl']; c = 0
    for i, (a0, a1) in enumerate(g['bounds']):
        if a1 <= n and i < g['lc']:
            apl[i] = a1 - a0; lc = i + 1
        else:
            apl[i] = max(0, n - a0); break
    return lc, apl, n

def summary(rel, T):
    r = load_run(rel)
    S = C = 0.0; acts = base = wasted = 0; lv = 0; full = 0
    for g in r['games']:
        lc, apl, n = at_time(g, T)
        s = official(g['nl'], g['base'], lc, apl); S += s
        C += 100 * sum(i + 1 for i in range(lc)) / (g['nl'] * (g['nl'] + 1) / 2)
        lv += lc; full += (lc == g['nl'])
        acts += sum(apl[:lc]); base += sum(g['base'][:lc])
        fa = failed_attempts(g)
        for i in range(lc):
            wasted += min(fa[i]['wasted'], apl[i])
    n = len(r['games'])
    return dict(S=S / n, C=C / n, ineff=(C - S) / n, levels=lv, full=full, ratio=acts / max(1, base), wasted=wasted / max(1, acts))

if __name__ == '__main__':
    Ts = [float(x) for x in sys.argv[1].split(',')]
    for rel in sys.argv[2:]:
        for T in Ts:
            s = summary(rel, T * 60)
            print(f"{rel:40s} @{T:>4.0f}m score {s['S']:6.2f} cap {s['C']:6.2f} ineff {s['ineff']:5.2f} levels {s['levels']:3d} full {s['full']:2d} actions/base {s['ratio']:.2f} failed-attempt share {s['wasted']:.1%}")
