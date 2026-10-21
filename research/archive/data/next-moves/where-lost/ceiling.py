"""Best-of-N ceiling per game, slot-time accounting, and per-level ladders."""
import statistics, json
from collections import defaultdict
from load import *
from decomp import REFS

STACK = ['v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers', 'v4-level-path-10x/v1']
ALLV4 = [r for r in REFS if r.startswith('v4-')] + ['v4-level-path-10x/v1']
runs = {}
for rel in set(ALLV4 + STACK):
    runs[rel] = load_run(rel)

def per_game(rels):
    d = defaultdict(list)
    for rel in rels:
        for g in runs[rel]['games']:
            d[g['key']].append((g['score'], g['lc'], g['nl']))
    return d

for name, rels in [('stack (10x x3 + level-path)', STACK), ('all v4 full runs (%d)' % len(ALLV4), ALLV4)]:
    d = per_game(rels)
    mean = sum(statistics.mean(x[0] for x in v) for v in d.values()) / 25
    best = sum(max(x[0] for x in v) for v in d.values()) / 25
    bestlv = {k: max(x[1] for x in v) for k, v in d.items()}
    print(f"{name}: mean {mean:.2f}, best-of-N per game {best:.2f} (gap {best-mean:.2f}); capability gap to 100: {100-best:.2f}")

# slot-time accounting on the stack runs
print('\nslot-time accounting (active minutes; gaps <=4 min between a game\'s actions):')
from decomp import ref_clear_times
need, reached = ref_clear_times()
for rel in STACK + ['v4-control/v1']:
    r = runs.get(rel) or load_run(rel)
    end = r['t_end']
    acc = defaultdict(float); parked = 0.0
    for g in r['games']:
        lts = level_times(g, end)
        for lt in lts:
            parked += lt['parked'] / 60
            if lt['cleared']:
                acc['cleared levels'] += lt['active'] / 60
            else:
                nd = need.get((g['key'], lt['level']), [])
                cls = 'frontier' if not nd else ('clock' if lt['active'] < statistics.median(nd) else 'stuck')
                acc['final level: ' + cls] += lt['active'] / 60
        if not g['at']:
            acc['never started'] += 0
    tot = sum(acc.values())
    print(f"  {rel:34s} active {tot:6.0f} min of {14*end/60:.0f} slot-min; " + ', '.join(f"{k} {v:.0f} ({v/tot:.0%})" for k, v in sorted(acc.items())) + f"; parked-gaps {parked:.0f}")

# per-level ladder: P(clear | reached) and median active minutes, all v4 runs
print('\nper-level ladder over all v4 full runs: reached / cleared / median active min to clear')
lad = defaultdict(lambda: defaultdict(lambda: [0, 0, []]))
for rel in ALLV4:
    r = runs[rel]; end = r['t_end']
    for g in r['games']:
        for lt in level_times(g, end):
            x = lad[g['key']][lt['level']]
            x[0] += 1
            if lt['cleared']:
                x[1] += 1; x[2].append(lt['active'] / 60)
for k in sorted(lad):
    s = []
    for lv in sorted(lad[k]):
        a, c, t = lad[k][lv]
        s.append(f"L{lv+1}:{c}/{a}" + (f"@{statistics.median(t):.0f}m" if t else ''))
    print(f"  {k}: " + '  '.join(s))
