"""Per game: minutes spent on each level (bench clock) for chosen runs; '*' marks the level open at the end."""
import json, sys
from collections import defaultdict
from runs_def import RUNS
runs = sys.argv[1].split(',')
games = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for g in games or []:
    for run in runs:
        rows = [json.loads(l) for l in open(RUNS[run] + '/score_timeline.jsonl') if l.strip()]
        rs = sorted([r for r in rows if r.get('kind') == 'game' and r['game'][:4] == g], key=lambda r: r['t_bench_s'])
        if not rs: continue
        n = rs[0]['n_levels']; t0 = rs[0]['t_bench_s'] / 60; done = 0; out = []
        for r in rs:
            while r['levels'] > done:
                done += 1; t = r['t_bench_s'] / 60; out.append(f'L{done}:{t - t0:.0f}'); t0 = t
        if done < n:
            out.append(f'L{done+1}*:{rs[-1]["t_bench_s"]/60 - t0:.0f}')
        print(f'{g} {run:6s} start {rs[0]["t_bench_s"]/60:5.1f}  ' + ' '.join(out))
    print()
