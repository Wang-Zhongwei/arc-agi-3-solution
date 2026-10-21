"""Restart-on-stall, simulated on the 12 full runs in token units.
Per (run, game, level): tokens spent on the level, cleared or not, tokens left in the run after it.
Trigger: tokens on the level >= max(FLOOR, MULT x median tokens of this game's earlier cleared
levels in this run) (L1: FLOOR x L1MULT). At a trigger the stalled attempt has B tokens of effort
left on this level (tokens until it cleared, or until the run ended). Continue: it clears iff it
did. Restart: a fresh attempt at the level, drawn from the other runs' attempts at the same level
(each a first attempt with prior-level memory): clears iff that attempt cleared within B tokens.
Reports expected levels gained and the level-clears traded away."""
import json, os, sys
from collections import defaultdict
from statistics import median
sys.path.insert(0, os.path.dirname(__file__))
from runs_def import RUNS

FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 60e3
MULT = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
L1MULT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
RESTART_COST = 20e3  # fresh opener prefill+first reply, counted against B

att = []  # dict(run, game, lv, tok, cleared, left_after) ; left_after = tokens this game spent after the level ended (for unfinished: 0)
for run, d in RUNS.items():
    per = defaultdict(list)
    for l in open(d + '/score_timeline.jsonl'):
        r = json.loads(l) if l.strip() else {}
        if r.get('kind') == 'game':
            per[r['game'][:4]].append(r)
    for g, rs in per.items():
        rs.sort(key=lambda r: r['t_bench_s'])
        n = rs[0]['n_levels']
        tok0, done = 0, 0
        for r in rs:
            while r['levels'] > done:
                done += 1
                att.append(dict(run=run, game=g, lv=done, tok=r['tokens'] - tok0, cleared=True))
                tok0 = r['tokens']
        if done < n:
            att.append(dict(run=run, game=g, lv=done + 1, tok=rs[-1]['tokens'] - tok0, cleared=False))

by_level = defaultdict(list)
for a in att:
    by_level[(a['game'], a['lv'])].append(a)
by_run_game = defaultdict(list)
for a in att:
    by_run_game[(a['run'], a['game'])].append(a)

events = []
for (run, g), seq in by_run_game.items():
    seq.sort(key=lambda a: a['lv'])
    for i, a in enumerate(seq):
        prev = [p['tok'] for p in seq[:i] if p['cleared']]
        thr = max(FLOOR, MULT * median(prev)) if prev else FLOOR * L1MULT
        if a['tok'] < thr:
            continue
        budget = a['tok'] - thr - RESTART_COST  # effort left on this level after the trigger
        others = [o for o in by_level[(g, a['lv'])] if o['run'] != run]
        if not others:
            continue
        p_fresh = sum(1 for o in others if o['cleared'] and o['tok'] <= budget) / len(others)
        events.append(dict(run=run, game=g, lv=a['lv'], thr=thr, tok=a['tok'], cleared=a['cleared'],
                           budget=budget, p_fresh=p_fresh, n_others=len(others), last=(a is seq[-1])))

cont = sum(e['cleared'] for e in events)
fresh = sum(e['p_fresh'] for e in events)
print(f'trigger max({FLOOR/1e3:.0f}k, {MULT}x median earlier cleared), L1 {FLOOR*L1MULT/1e3:.0f}k: '
      f'{len(events)} events over 12 runs ({len(events)/12:.1f}/run)')
print(f'  continue clears {cont} ({cont/len(events):.2f}); restart clears (expected) {fresh:.1f} ({fresh/len(events):.2f}); '
      f'net {fresh - cont:+.1f} levels over 12 runs ({(fresh - cont)/12:+.2f}/run)')
# split: stalls that never cleared vs ones that cleared later
nc = [e for e in events if not e['cleared']]
yc = [e for e in events if e['cleared']]
print(f'  never-cleared stalls {len(nc)}: restart expected clears {sum(e["p_fresh"] for e in nc):.1f}')
print(f'  later-cleared stalls {len(yc)}: restart expected clears {sum(e["p_fresh"] for e in yc):.1f} (continue: {len(yc)})')
if '-v' in sys.argv:
    for e in sorted(events, key=lambda e: (e['game'], e['lv'], e['run'])):
        print(f"  {e['run']:6s} {e['game']} L{e['lv']} thr {e['thr']/1e3:5.0f}k tok {e['tok']/1e3:5.0f}k "
              f"{'clr' if e['cleared'] else 'NO '} budget {e['budget']/1e3:5.0f}k p_fresh {e['p_fresh']:.2f} (n={e['n_others']})")
