"""Per game in run R: levels done/n, points at stake on the open level and beyond, minutes on the open level, start time; plus wins across all runs."""
import json, lzma, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs_def import RUNS
from decomp import load
R = sys.argv[1]
allruns = [r for r in RUNS if r not in ('comb',)]
wins = {}; best = {}
for r in allruns:
    b = load(RUNS[r])
    if not b: continue
    for g in b['game_runs']:
        k = g['game_id'][:4]; d = int(g['levels_completed']); n = int(g['number_of_levels'])
        wins[k] = wins.get(k, 0) + (d == n); best[k] = max(best.get(k, 0), d)
rows = [json.loads(l) for l in open(RUNS[R] + '/score_timeline.jsonl') if l.strip()]
b = load(RUNS[R])
print(f'game  done/n  open-level-min  next-level-weight%  rest-weight%  best-ever  wins/{len(allruns)}  tokens')
out = []
for g in b['game_runs']:
    k = g['game_id'][:4]; d = int(g['levels_completed']); n = int(g['number_of_levels'])
    rs = sorted([x for x in rows if x.get('kind') == 'game' and x['game'][:4] == k], key=lambda x: x['t_bench_s'])
    t_last = None
    for x in rs:
        if x['levels'] == d: t_last = x['t_bench_s'] if t_last is None else t_last
    t_open = (rs[-1]['t_bench_s'] - t_last) / 60 if (rs and t_last is not None and d < n) else 0
    W = n * (n + 1) / 2
    nxt = (d + 1) / W * 100 if d < n else 0; rest = sum(range(d + 1, n + 1)) / W * 100
    tok = rs[-1]['tokens'] if rs else 0
    out.append((rest, f'{k}  {d}/{n}  {t_open:5.0f}  {nxt:5.1f}  {rest:5.1f}  {best[k]}  {wins[k]}  {tok/1e3:.0f}k'))
for _, s in sorted(out, reverse=True): print(s)
