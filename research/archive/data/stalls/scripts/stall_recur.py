"""How many of the full runs stall on each (game, level): >= 60k tokens or >= 45 min, cleared or not."""
import json
from collections import defaultdict, Counter
from runs_def import RUNS
cnt = Counter()
for run, d in RUNS.items():
    rows = [json.loads(l) for l in open(d + '/score_timeline.jsonl') if l.strip()]
    per = defaultdict(list)
    for r in rows:
        if r.get('kind') == 'game': per[r['game'][:4]].append(r)
    for g, rs in per.items():
        rs.sort(key=lambda r: r['t_bench_s']); t0, k0, done = rs[0]['t_bench_s'], 0, 0
        for r in rs:
            if r['levels'] > done:
                if r['tokens'] - k0 >= 60000 or (r['t_bench_s'] - t0) / 60 >= 45: cnt[(g, done + 1)] += 1
                done, t0, k0 = r['levels'], r['t_bench_s'], r['tokens']
        if done < rs[0]['n_levels'] and (rs[-1]['tokens'] - k0 >= 60000 or (rs[-1]['t_bench_s'] - t0) / 60 >= 45):
            cnt[(g, done + 1)] += 1
print(len(RUNS), 'runs'); print(' '.join(f'{g}L{l}:{n}' for (g, l), n in cnt.most_common(20)))
