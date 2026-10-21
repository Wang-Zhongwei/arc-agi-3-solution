# Final levels, tokens and levels per 1M generated tokens from score_timeline.jsonl (last row per game).
import json, sys, os
for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(os.path.join(p, 'score_timeline.jsonl'))]
    g = [r for r in rows if r.get('kind') == 'game']
    last = {}
    for r in g: last[r['game']] = r
    lv = sum(r['levels'] for r in last.values()); tok = sum(r['tokens'] for r in last.values())
    acts = sum(r['actions'] for r in last.values()); tmax = max(r['t_bench_s'] for r in g)
    sc = sum(r['score'] for r in last.values()) / 25
    print(f'{p.split("runs/kaggle/")[-1].split("runs-out/")[-1]:32s} score {sc:6.2f} levels {lv:4d} tokens {tok/1e6:5.2f}M  levels/1M tok {lv/(tok/1e6):5.1f}  tok/s {tok/tmax:5.0f}  actions {acts}')
