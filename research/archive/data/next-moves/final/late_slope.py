# Score at earlier wall-clock cut-offs vs the final, per run: how much do the last 10-20% of a run's minutes add?
import json, sys, glob, os
root = sys.argv[1]
runs = sys.argv[2:]
for rel in runs:
    p = os.path.join(root, rel, 'score_timeline.jsonl')
    rows = [json.loads(l) for l in open(p)]
    g = [r for r in rows if r.get('kind') == 'game']
    games = sorted({r['game'] for r in g})
    tmax = max(r['t_bench_s'] for r in g)
    def at(t):
        last = {}
        nl = {}
        for r in g:
            if r['t_bench_s'] <= t:
                last[r['game']] = r['score']; nl[r['game']] = (r['levels'], r['n_levels'])
        sc = sum(last.get(k, 0.0) for k in games) / 25
        wins = sum(1 for k, (a, b) in nl.items() if a == b)
        return sc, wins
    out = []
    for frac in (0.8, 0.85, 0.9, 0.95, 1.0):
        s, w = at(tmax * frac)
        out.append(f'{frac:.2f}T={tmax*frac/60:.0f}m {s:.2f} (wins {w})')
    print(rel, len(games), 'games |', ' | '.join(out))
