"""Decompose 100 - score per game into: cleared-level inefficiency (c), failed attempts (d), stuck level, levels beyond.
Usage: python3 decomp.py RUN [RUN...]  (relative to om/runs/kaggle)"""
import sys, statistics, json
from collections import defaultdict
from load import *

REFS = ['v4-control/v1', 'v4-user-combo-10x/v1', 'v4-user-combo-10x/v2-snowiers', 'v4-user-combo-10x/v3-snowiers',
        'v4-user-combo/v1', 'v4-user-combo-13x/v1', 'v4-sched-curriculum/v1', 'v4-user-combo-10x-accept05/v1',
        'v4-hats/v1', 'v4-user-combo-16x/v1', 'v4-uc-11x/v1-snowiers', 'v4-uc-18s-64k/v1', 'v4-stack-v1/v1',
        'fz-ledger-on-trim/v1', 'fz-ledger-on-trim-r2/v1']

def ref_clear_times():
    need = defaultdict(list)   # (game, level) -> active seconds to clear
    reached = defaultdict(int)
    for rel in REFS:
        try:
            r = load_run(rel)
        except Exception as e:
            print('skip', rel, e, file=sys.stderr); continue
        end = r['t_end'] or 7260
        for g in r['games']:
            for lt in level_times(g, end):
                reached[(g['key'], lt['level'])] += 1
                if lt['cleared']:
                    need[(g['key'], lt['level'])].append(lt['active'])
    return need, reached

def wshare(nl, idxs):
    W = nl * (nl + 1) / 2
    return 100 * sum(i + 1 for i in idxs) / W

REACHED = {}

def decomp_game(g, end, need):
    nl, base, apl, lc = g['nl'], g['base'], g['apl'], g['lc']
    S = official(nl, base, lc, apl)
    C = wshare(nl, range(lc))
    fa = failed_attempts(g)
    apl_nofail = [apl[i] - (fa[i]['wasted'] if (i in fa and i < lc and fa[i]['wasted'] < apl[i]) else 0) for i in range(nl)]
    S_nofail = official(nl, base, lc, apl_nofail)
    apl_15 = [min(apl[i], int(1.5 * base[i] + 0.999)) if i < lc else apl[i] for i in range(nl)]
    S_cap15 = official(nl, base, lc, apl_15)
    apl_10 = [min(apl[i], base[i]) if i < lc else apl[i] for i in range(nl)]
    S_cap10 = official(nl, base, lc, apl_10)
    row = dict(game=g['key'], nl=nl, lc=lc, S=S, C=C, ineff=C - S, d_fail=S_nofail - S, S_cap15=S_cap15, S_cap10=S_cap10,
               stuck_w=0.0, beyond_w=0.0, cls='won', act_min=0, park_min=0, need_min=None, start_min=(g['off'] or 0) / 60,
               n_go=sum(v['n_go'] for v in fa.values()), n_reset=sum(v['n_reset'] for v in fa.values()),
               wasted_actions_cleared=sum(fa[i]['wasted'] for i in fa if i < lc),
               actions_cleared=sum(apl[:lc]), base_cleared=sum(base[:lc]),
               ratio_levels=[(apl[i], base[i]) for i in range(lc)])
    if lc < nl:
        row['stuck_w'] = wshare(nl, [lc]); row['beyond_w'] = wshare(nl, range(lc + 1, nl))
        lts = level_times(g, end)
        cur = lts[-1] if lts and lts[-1]['level'] == lc else None
        tot_park = sum(x['parked'] for x in lts)
        row['park_total_min'] = tot_park / 60
        if not g['at']:
            row['cls'] = 'not_started'
        else:
            row['act_min'] = cur['active'] / 60 if cur else 0
            row['park_min'] = cur['parked'] / 60 if cur else 0
            nd = need.get((g['key'], lc), [])
            if not nd:
                row['cls'] = 'frontier'
            else:
                med = statistics.median(nd); row['need_min'] = med / 60; row['n_ref_clears'] = len(nd)
                rate = len(nd) / max(1, REACHED.get((g['key'], lc), len(nd)))
                row['clear_rate'] = rate
                if rate < 0.5:
                    row['cls'] = 'wall'
                else:
                    row['cls'] = 'clock' if cur and cur['active'] < med else 'stuck'
    return row

if __name__ == '__main__':
    need, reached = ref_clear_times()
    REACHED.update(reached)
    out = {}
    for rel in sys.argv[1:]:
        r = load_run(rel)
        end = r['t_end'] or max((max(g['at']) if g['at'] else 0) for g in r['games'])
        rows = [decomp_game(g, end, need) for g in r['games']]
        out[rel] = rows
        n = len(rows)
        S = sum(x['S'] for x in rows) / n
        print(f"\n=== {rel}  mean {S:.2f}  (end {end/60:.1f} min)")
        print(f"{'game':5s} {'lv':>5s} {'score':>6s} {'cap':>6s} {'c_ineff':>7s} {'d_fail':>6s} {'stuck':>6s} {'beyond':>6s} cls        act/need/park(min) start  @1.5x  @1.0x  go/rs  acts/base(cleared)")
        for x in sorted(rows, key=lambda x: -(100 - x['S'])):
            nd = f"{x['need_min']:.0f}" if x['need_min'] is not None else '-'
            print(f"{x['game']:5s} {x['lc']}/{x['nl']:<3d} {x['S']:6.1f} {x['C']:6.1f} {x['ineff']:7.1f} {x['d_fail']:6.1f} {x['stuck_w']:6.1f} {x['beyond_w']:6.1f} {x['cls']:10s} {x['act_min']:4.0f}/{nd:>3s}/{x['park_min']:3.0f}   {x['start_min']:4.0f} {x['S_cap15']:6.1f} {x['S_cap10']:6.1f}  {x['n_go']}/{x['n_reset']}  {x['actions_cleared']}/{x['base_cleared']}")
        tot = defaultdict(float)
        for x in rows:
            tot['ineff'] += x['ineff']; tot['d_fail'] += x['d_fail']
            tot['stuck_' + x['cls']] += x['stuck_w']; tot['beyond_' + x['cls']] += x['beyond_w']
            tot['gain_cap15'] += x['S_cap15'] - x['S']; tot['gain_cap10'] += x['S_cap10'] - x['S']
        print('run-points (sum/25):', {k: round(v / n, 2) for k, v in sorted(tot.items())})
        print('check: S + ineff + stuck + beyond =', round((sum(x['S'] for x in rows) + tot['ineff'] + sum(x['stuck_w'] + x['beyond_w'] for x in rows)) / n, 2))
    json.dump(dict(out=out, need={f'{k[0]}|{k[1]}': v for k, v in need.items()}, reached={f'{k[0]}|{k[1]}': v for k, v in reached.items()}),
              open('decomp.json', 'w'), default=str)
