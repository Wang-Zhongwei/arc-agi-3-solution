"""Level durations (bench minutes) per run/game/level from score timelines, and what a stall at
T minutes means: P(clear later in the same attempt) vs how fast other runs cleared that level."""
import json, sys, glob, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from runs_def import RUNS

END = 121.0
T = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
levels = []  # (run, game, level_idx(1-based), start, end or None, cleared)
for run, d in RUNS.items():
    rows = [json.loads(l) for l in open(d + '/score_timeline.jsonl') if l.strip()]
    per = defaultdict(list)
    for r in rows:
        if r.get('kind') == 'game':
            per[r['game'][:4]].append(r)
    for g, rs in per.items():
        rs.sort(key=lambda r: r['t_bench_s'])
        n = rs[0]['n_levels']
        start, done = rs[0]['t_bench_s'] / 60, 0
        last_t = rs[-1]['t_bench_s'] / 60
        finished = rs[-1]['state'] in ('finished', 'WIN', 'won') or rs[-1]['levels'] >= n
        for r in rs:
            while r['levels'] > done:
                done += 1
                t = r['t_bench_s'] / 60
                levels.append((run, g, done, start, t, True))
                start = t
        if done < n:
            levels.append((run, g, done + 1, start, None, False))

json.dump(levels, open(os.path.join(os.path.dirname(__file__), 'stall_levels.json'), 'w'))
cleared = [l for l in levels if l[5]]
dur = lambda l: (l[4] if l[4] is not None else END) - l[3]
# 1. How often does a level run past T minutes, and how does it end?
past = [l for l in levels if dur(l) >= T]
print(f'levels total {len(levels)}, cleared {len(cleared)}; past {T:.0f} min: {len(past)}, of which cleared later {sum(l[5] for l in past)}')
# 2. For each stall past T: the same level in other runs (that reached it) -- share cleared, median minutes
by = defaultdict(list)
for l in levels:
    by[(l[1], l[2])].append(l)
rows = []
for l in past:
    others = [o for o in by[(l[1], l[2])] if o[0] != l[0]]
    oc = [dur(o) for o in others if o[5]]
    within20 = sum(1 for o in others if o[5] and dur(o) <= 20)
    rows.append((l, len(others), len(oc), within20, sorted(oc)))
print(f"\n{'run':6s} {'game':5s} {'L':>2s} {'start':>5s} {'dur':>5s} {'clr':>4s} | others reached / cleared / cleared<=20m | other clear mins")
for l, n, c, w20, oc in sorted(rows, key=lambda x: (x[0][1], x[0][2], x[0][0])):
    print(f'{l[0]:6s} {l[1]:5s} {l[2]:2d} {l[3]:5.0f} {dur(l):5.0f} {"yes" if l[5] else "NO":>4s} | {n:2d} / {c:2d} / {w20:2d} | {" ".join(f"{x:.0f}" for x in oc[:10])}')
# 3. Pooled: for stalls past T on levels others usually clear fast
easy = [(l, n, c, w20) for l, n, c, w20, _ in rows if n >= 3 and w20 / n >= 0.5]
print(f'\nstalls past {T:.0f}m on levels that >=50% of other runs cleared within 20m: {len(easy)}; '
      f'cleared later in the same attempt: {sum(l[5] for l, *_ in easy)}; '
      f'runs-games affected: {len({(l[0], l[1]) for l, *_ in easy})}')
