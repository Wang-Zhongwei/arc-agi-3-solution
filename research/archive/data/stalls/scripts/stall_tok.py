"""Per run: tokens and minutes on stalled levels (>=60k tokens or >=45 min on one level, open or cleared)."""
import json
from collections import defaultdict
from runs_def import RUNS
for run, d in RUNS.items():
    rows = [json.loads(l) for l in open(d + '/score_timeline.jsonl') if l.strip()]
    per = defaultdict(list)
    for r in rows:
        if r.get('kind') == 'game': per[r['game'][:4]].append(r)
    lv = []
    for g, rs in per.items():
        rs.sort(key=lambda r: r['t_bench_s'])
        t0, k0, done = rs[0]['t_bench_s'], 0, 0
        for r in rs:
            if r['levels'] > done:
                lv.append((g, done + 1, r['tokens'] - k0, (r['t_bench_s'] - t0) / 60, True))
                done, t0, k0 = r['levels'], r['t_bench_s'], r['tokens']
        if done < rs[0]['n_levels']:
            lv.append((g, done + 1, rs[-1]['tokens'] - k0, (rs[-1]['t_bench_s'] - t0) / 60, False))
    st = [x for x in lv if x[2] >= 60000 or x[3] >= 45]
    tot = sum(x[2] for x in lv)
    print(f"{run:6s} total {tot/1e6:4.2f}M  stalls {len(st):2d} ({sum(1 for x in st if not x[4])} open)  "
          f"stall tok {sum(x[2] for x in st)/1e6:4.2f}M  stall min {sum(x[3] for x in st):5.0f}  "
          + ' '.join(f"{g}L{l}{'' if c else '*'}:{m:.0f}" for g, l, k, m, c in sorted(st, key=lambda x: -x[3])[:8]))
